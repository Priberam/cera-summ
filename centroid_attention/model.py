from functools import partial

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from centroid_attention.model_utils import cosine_loss, separate_per_document
from ROUGE.rouge_script import run_ROUGE
from summarizer.summaries import sentence_selection
from summarizer.utils import (
    dict_multi_docs_lines,
    eliminate_null_embeddings,
    process_data_dicts,
)

# Create a partial function from the Beam Search Deep function
Part_DeepBeamSrc = partial(
    sentence_selection,
    sentences="n_first",
    n=10,
    beam_width=3,
    counter_limit=3,
    alpha=0,
)


class CentroidEstimationModel(nn.Module):
    """
    Functioning:
        The first FC unit maps to a space with num_heads dimensions
        to compute attention scores for multiple heads. The input data
        is copied num_heads times and multiplied by the different
        attention scores. In this process, the samples of each head are
        put together as different elements of the batch, ultimately
        increasing the batch size during these operations. The embeddings
        resultant from the attention have embed_dim*num_heads size and
        are fed to the last FC unit to estimate a centroid. Additionaly,
        interpolation with the unsupervised centroid can be performed.

        For this model there are also several additional options like
        concatenating the input data with the cluster mean pool or the
        cosine similarity between each sentence embedding and the cluster
        mean pool. We can also use residual connections before and after
        computing the centroid. There is
        also the option of normalizing the input data to have unit norm
        and layernorms can also be applied to the sentence embeddings and
        to the data flowing through the model before the last FC layer
        or/and after the last FC layer also.

    """

    def __init__(
        self,
        embed_dim=512,
        num_positional_embeddings=30,
        num_heads=4,
        dropout=0,
        norm_1=True,
        input_layernorm=True,
        beforeFC_output_layernorm=False,
        afterFC_output_layernorm=True,
        concat_meanpool=True,
        concat_cosine=True,
        use_residual=True,
        use_bias=True,
        interpolation=False,
    ):
        super().__init__()

        # Number of heads to be used in the model
        self.num_heads = num_heads

        # To normalize the data (l2-norm=1)
        self.norm_1 = norm_1

        # Apply a normalization layer to the input data
        self.input_layernorm = input_layernorm

        # Apply a normalization layer before the last FC layer that estimates
        # the centroid
        self.beforeFC_output_layernorm = beforeFC_output_layernorm
        # Apply a normalization layer to the predicted centroid
        self.afterFC_output_layernorm = afterFC_output_layernorm

        # Select what is going to be concatenated with the embeddings
        self.concat_meanpool = concat_meanpool
        self.concat_cosine = concat_cosine

        # Select if residual connections are used or not
        self.use_residual = use_residual

        # Activate or deactivate the bias on the first FC
        self.use_bias = use_bias

        # Apply an interpolation on the att. model output
        self.interpolation = interpolation

        # Dropout layer
        self.drop = nn.Dropout(p=dropout)

        # First normalization layer
        if self.input_layernorm:
            self.layernorm1 = nn.LayerNorm(
                normalized_shape=embed_dim,  # elementwise_affine=False
            )

        # Positional embeddings
        self.pos = nn.Embedding(
            num_embeddings=num_positional_embeddings + 1,
            embedding_dim=embed_dim,
            padding_idx=num_positional_embeddings,
        )

        # The first FC layer dimensions depends on what is concatenated
        # with the embeddings (if anything is concatenated at all)
        if self.concat_meanpool or self.concat_cosine:
            if self.concat_meanpool and self.concat_cosine:
                self.fc1 = nn.Sequential(
                    nn.Linear(
                        in_features=embed_dim * 2 + 1,
                        out_features=embed_dim,
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        in_features=embed_dim, out_features=self.num_heads
                    ),
                )
            elif self.concat_meanpool:
                self.fc1 = nn.Sequential(
                    nn.Linear(
                        in_features=embed_dim * 2,
                        out_features=embed_dim,
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        in_features=embed_dim, out_features=self.num_heads
                    ),
                )
            elif self.concat_cosine:
                self.fc1 = nn.Sequential(
                    nn.Linear(
                        in_features=embed_dim + 1,
                        out_features=embed_dim,
                    ),
                    nn.Tanh(),
                    nn.Linear(
                        in_features=embed_dim, out_features=self.num_heads
                    ),
                )
        else:
            if self.use_bias:
                self.fc1 = nn.Linear(
                    in_features=embed_dim, out_features=self.num_heads
                )
            else:
                self.fc1 = nn.Linear(
                    in_features=embed_dim,
                    out_features=self.num_heads,
                    bias=False,
                )

        # Normalization layer to be applied before or after estimating
        # the centroid
        if self.beforeFC_output_layernorm:
            self.layernorm3 = nn.LayerNorm(
                normalized_shape=embed_dim * self.num_heads,
            )
        elif self.afterFC_output_layernorm:
            self.layernorm3 = nn.LayerNorm(normalized_shape=embed_dim)

        # Residual connections
        if self.use_residual:
            self.layernorm2 = nn.LayerNorm(embed_dim * self.num_heads)
            self.layernorm4 = nn.LayerNorm(embed_dim)

        # Attention model output layer (predict the centroid)
        self.fc2 = nn.Linear(
            in_features=embed_dim * self.num_heads, out_features=embed_dim
        )

        # Model to predict a vector of alphas that control the interpolation
        if self.interpolation:
            self.alpha_nn = nn.Sequential(
                nn.Linear(in_features=embed_dim * 2, out_features=embed_dim),
                nn.ReLU(),
                nn.Linear(in_features=embed_dim, out_features=embed_dim),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x,
        attention_mask,
        order,
        num_docs,
        docs_weights,
        clusters_centroids,
    ):
        # x.shape = (batch_size, max_S, embed_size)
        # attention_mask.shape = (batch_size, max_S)
        # order.shape = (batch_size, max_S)
        # num_docs.shape = (batch_size)
        # docs_weights = (batch_size, max_S)
        # clusters_centroids = (batch_size, embed_size)

        # Fetch the positional embeddings
        # (batch_size, max_S, embed_size)
        pos_encodings = self.pos(order)

        # Normalize the input data (l2-norm = 1)
        if self.norm_1:
            # Use eps to avoid dividing by 0
            x = torch.nn.functional.normalize(x.clone(), dim=2, eps=1e-8)

        # Pass the input data through a normalization layer
        if self.input_layernorm:
            # (batch_size, max_S, embed_size)
            x = self.layernorm1(x)

        # Sum the positional embeddings to the input data
        # (batch_size, max_S, embed_size)
        x_positions = x + pos_encodings

        # Add dropout to the model input
        # (batch_size, max_S, embed_size)
        x_positions = self.drop(x_positions)

        # Prepare the input for the first FC layer
        # (batch_size, max_S, embed_size)
        fc1_input = x_positions

        # Compute the cluster meanpool vector
        if self.concat_meanpool or self.concat_cosine:
            # (batch_size, max_S, embed_size)
            mean_pool = x_positions * docs_weights.unsqueeze(-1)
            # (batch_size, 1, embed_size)
            mean_pool = torch.sum(mean_pool, dim=1, keepdim=True)
            # (batch_size, 1, embed_size)
            mean_pool = mean_pool / num_docs.unsqueeze(-1).unsqueeze(-1)

        # Concatenate the meanpool vector with each input embedding
        if self.concat_meanpool:
            # (batch_size, 1, embed_size) -> # (batch_size, max_S, embed_size)
            aux_mean_pool = mean_pool.repeat(1, fc1_input.size(1), 1)
            # (batch_size, max_S, embed_size) ->
            # (batch_size, max_S, embed_size*2)
            fc1_input = torch.cat((fc1_input, aux_mean_pool), dim=2)

        # Concatenate the cosine sim. between each input embedding and the
        # mean pool with the input embedding
        if self.concat_cosine:
            # (batch_size, max_S)
            cos_similarity = F.cosine_similarity(mean_pool, x_positions, dim=2)
            # (batch_size, max_S,1)
            cos_similarity = cos_similarity.unsqueeze(-1)

            # (batch_size, max_S, embed_size*2+1)
            # if self.concat_meanpool = True
            # (batch_size, max_S, embed_size+1)
            # if self.concat_meanpool = False
            fc1_input = torch.cat((fc1_input, cos_similarity), dim=2)

        # (batch_size, max_S, embed_size*2+1) -> (batch_size, max_S, num_heads)
        #  if self.concat_meanpool = True and self.concat_cosine = True or
        # (batch_size, max_S, embed_size*2) -> (batch_size, max_S, num_heads)
        #  if self.concat_meanpool = True and self.concat_cosine = False or
        # (batch_size, max_S, embed_size+1) -> (batch_size, max_S, num_heads)
        #  if self.concat_meanpool = False and self.concat_cosine = True or
        # (batch_size, max_S, embed_size) -> (batch_size, max_S, num_heads)
        #  if self.concat_meanpool = False and self.concat_cosine = False
        Z = self.fc1(fc1_input)

        # Reshape the attention mask
        # (batch_size, max_S) -> (batch_size, max_S, num_heads)
        attention_mask = attention_mask.unsqueeze(2).repeat(
            1, 1, self.num_heads
        )

        # Masked logits (batch_size, max_S, num_heads)
        Z.masked_fill_(attention_mask, -float("inf"))

        # Attention weights (batch_size, max_S, num_heads)
        A = torch.softmax(Z, dim=1)

        # Treat the attention vector as different batch elements
        # (batch_size, max_S, num_heads) -> (batch_size*num_heads, max_S, 1)
        A = A.reshape(A.size(0) * A.size(2), A.size(1), 1)

        # Repeat the input data num_heads times
        # (batch_size, max_S, embed_size) ->
        # (batch_size, num_heads, max_S, embed_size)
        aux_x = x.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        # (batch_size, num_heads, max_S, embed_size) ->
        # (batch_size*num_heads, max_S, embed_size)
        aux_x = aux_x.reshape(
            aux_x.size(0) * aux_x.size(1), aux_x.size(2), aux_x.size(3)
        )

        # (batch_size*num_heads, max_S, embed_size) ->
        # (batch_size, embed_size*num_heads)
        H = torch.bmm(aux_x.transpose(1, 2), A).reshape(Z.size(0), -1)

        # Residual Connection after the attention
        if self.use_residual:
            # Mean pool x
            # (batch_size, max_S, embed_size)
            mean_pool_x = x * docs_weights.unsqueeze(-1)
            # (batch_size, 1, embed_size)
            mean_pool_x = torch.sum(mean_pool_x, dim=1, keepdim=True)
            # (batch_size, 1, embed_size)
            mean_pool_x = mean_pool_x / num_docs.unsqueeze(-1).unsqueeze(-1)

            # RESHAPE mean_pool_x
            # (batch_size, 1, embed_size) ->
            # (batch_size,num_heads, 1, embed_size)
            mean_pool_x = mean_pool_x.unsqueeze(1).repeat(
                1, self.num_heads, 1, 1
            )
            # (batch_size,num_heads, 1, embed_size) ->
            # (batch_size, 1, embed_size*num_heads)
            mean_pool_x = mean_pool_x.reshape(
                mean_pool_x.size(0),
                mean_pool_x.size(2),
                mean_pool_x.size(3) * mean_pool_x.size(1),
            )

            # (batch_size, 1, embed_size*num_heads) ->
            # (batch_size, embed_size*num_heads)
            mean_pool_x = mean_pool_x.squeeze(1)

            # (batch_size, embed_size*num_heads)
            H = H + mean_pool_x

            # Apply layer normalization
            # (batch_size, embed_size*num_heads)
            H = self.layernorm2(H)

        # Add dropout to the masked embedding
        H = self.drop(H)

        # Pass the predicted centroid through a normalization layer
        if self.beforeFC_output_layernorm:
            # (batch_size, embed_size*num_heads)
            H = self.layernorm3(H)

        #  (batch_size, embed_size*num_heads) -> (batch_size, embed_size)
        pred = self.fc2(H)

        # Pass the predicted centroid through a normalization layer
        if self.afterFC_output_layernorm:
            #  (batch_size, embed_size*num_heads) -> (batch_size, embed_size)
            pred = self.layernorm3(pred)

        # Consider a residual connection applide to the model output
        if self.use_residual:
            # (batch_size, embed_size* num_heads) ->
            # (batch_size, num_heads, embed_size)
            H = H.reshape(H.size(0), self.num_heads, pred.size(1))
            # (batch_size, num_heads, embed_size) ->
            # (batch_size, embed_size) (Average on num_heads)
            H = torch.mean(H, dim=1)
            # Add residual
            pred = pred + H
            # Layer normalization
            pred = self.layernorm4(pred)

        # Perform an interpolation between the model output and the
        # clusters' centroid
        if self.interpolation:
            # Learn alpha
            # (batch_size, embed_size) ->(batch_size, embed_size*2)
            alpha_model_in = torch.cat((pred, clusters_centroids), dim=1)
            # (batch_size, embed_size*2) -> (batch_size, embed_size)
            alpha = self.alpha_nn(alpha_model_in)

            # Obtain the new prediction through interpolation
            # (batch_size, embed_size)
            pred = alpha * pred + (1 - alpha) * clusters_centroids

        return pred


class LitCentroidEstimationModel(pl.LightningModule):
    def __init__(
        self,
        embed_dim,
        num_positional_embeddings,
        num_heads,
        dropout,
        norm_1,
        input_layernorm,
        beforeFC_output_layernorm,
        afterFC_output_layernorm,
        concat_meanpool,
        concat_cosine,
        use_residual,
        use_bias,
        interpolation,
        batch_size,
        learning_rate,
        use_scheduler,
        scheduler_step_size,
        scheduler_gamma,
        loss_type,
        budget=230,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.attention_model = CentroidEstimationModel(
            embed_dim,
            num_positional_embeddings,
            num_heads,
            dropout,
            norm_1,
            input_layernorm,
            beforeFC_output_layernorm,
            afterFC_output_layernorm,
            concat_meanpool,
            concat_cosine,
            use_residual,
            use_bias,
            interpolation,
        )
        # Save the outputs at each validation step
        # so that we can compute the ROUGE scores
        # on validation_epoch_end()
        self.validation_step_outputs = {
            "centroid": [],
            "target": [],
            "multi_docs_lines": [],
            "doc_fts": [],
        }
        # Keep track of the best R2-R (for logging purposes)
        self.best_R2_R = 0

        torch.set_float32_matmul_precision("high")

    def forward(
        self,
        x,
        attention_mask,
        order,
        num_docs,
        docs_weights,
        clusters_centroids,
    ):
        return self.attention_model(
            x,
            attention_mask,
            order,
            num_docs,
            docs_weights,
            clusters_centroids,
        )

    def training_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        clusters_centroids = batch["clusters_centroids"]
        attention_mask = batch["attention_mask"]
        order = batch["order"]
        num_docs = batch["num_docs"]
        docs_weights = batch["docs_weights"]

        # Prediction
        pred_centroid = self.attention_model(
            x,
            attention_mask,
            order,
            num_docs,
            docs_weights,
            clusters_centroids,
        )

        # Loss
        if self.hparams.loss_type == "cosine":
            loss = cosine_loss(output=pred_centroid, target=y)
        elif self.hparams.loss_type == "mse":
            loss = F.mse_loss(pred_centroid, y)
        else:
            raise NotImplementedError

        # Logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["x"]  # (batch_size, max_S,embed_size)
        y = batch["y"]  # (batch_size, 1,embed_size)
        clusters_centroids = batch["clusters_centroids"]
        attention_mask = batch["attention_mask"]  # (batch_size, max_S)
        order = batch["order"]  # (batch_size, max_S)
        num_docs = batch["num_docs"]  # (batch_size)
        docs_weights = batch["docs_weights"]  # (batch_size, max_S)
        docs_length = batch["docs_length"]
        multi_docs_lines = batch["multi_docs_lines"]
        target = batch["target"]

        # Prediction
        pred_centroid = self.attention_model(
            x,
            attention_mask,
            order,
            num_docs,
            docs_weights,
            clusters_centroids,
        )

        # Loss
        if self.hparams.loss_type == "cosine":
            loss = cosine_loss(output=pred_centroid, target=y)
        elif self.hparams.loss_type == "mse":
            loss = F.mse_loss(pred_centroid, y)
        else:
            raise NotImplementedError

        # Validation loss
        self.log("val_loss", loss, batch_size=self.hparams.batch_size)

        # Update self.validation_step_outputs
        for i in range(x.shape[0]):
            # Predicted centroid
            centroid = pred_centroid[i].detach().cpu().numpy()

            # Separate the clusters' features per document
            doc_fts, _ = separate_per_document(
                x[i].to("cpu"), order[i], docs_length[i]
            )

            # Transform multi_docs_lines into a list of lists of
            # dictionaries
            multi_docs_lines[i] = dict_multi_docs_lines(multi_docs_lines[i])

            # Process the data before the summarization algorithm (no
            # repeated sentences and no sentences exceeding the budget)
            multi_docs_lines[i], doc_fts, _ = process_data_dicts(
                multi_docs_lines[i], doc_fts, self.hparams.budget
            )

            # Save the results of a validation step
            self.validation_step_outputs["centroid"].append(centroid)
            self.validation_step_outputs["target"].append(target[i])
            self.validation_step_outputs["multi_docs_lines"].append(
                multi_docs_lines[i]
            )
            self.validation_step_outputs["doc_fts"].append(doc_fts)

        return loss

    def on_validation_epoch_end(self):
        summaries = []
        targets = []

        # Iterate through the saved data during the validation steps,
        # compute the summaries from this data and evaluate them
        # via ROUGE
        for i in tqdm(
            range(len(self.validation_step_outputs["centroid"])),
            desc="Building summaries...",
        ):
            centroid = self.validation_step_outputs["centroid"][i]
            multi_docs_lines = self.validation_step_outputs[
                "multi_docs_lines"
            ][i]
            doc_features = self.validation_step_outputs["doc_fts"][i]

            targets.append([self.validation_step_outputs["target"][i]])

            # Eliminate cases where embeddings are null
            multi_docs_lines, doc_features = eliminate_null_embeddings(
                multi_docs_lines, doc_features
            )

            summary = Part_DeepBeamSrc(
                multi_docs_lines=multi_docs_lines,
                doc_features=doc_features,
                centroid=centroid,
                budget=self.hparams.budget,
            )

            summary = " ".join([item["sentence"] for item in summary])
            summaries.append(summary)

        # Compute Rouge Scores using the PERL script
        metrics = run_ROUGE(
            model_summaries=targets,
            system_summaries=summaries,
            budget=self.hparams.budget,
        )

        # Log the R2-R values
        self.log(
            "R2_R",
            metrics["rouge_2_recall"],
            batch_size=self.hparams.batch_size,
        )

        # Log the best R2-R
        if self.best_R2_R < metrics["rouge_2_recall"]:
            self.best_R2_R = metrics["rouge_2_recall"]

        # Log the best R2-R so far
        self.log(
            "Best_R2_R",
            self.best_R2_R,
            batch_size=self.hparams.batch_size,
        )
        # free memory
        self.validation_step_outputs["centroid"].clear()
        self.validation_step_outputs["target"].clear()
        self.validation_step_outputs["multi_docs_lines"].clear()
        self.validation_step_outputs["doc_fts"].clear()

    def test_step(self, batch, batch_idx):
        x = batch["x"]
        y = batch["y"]
        clusters_centroids = batch["clusters_centroids"]
        attention_mask = batch["attention_mask"]
        order = batch["order"]
        num_docs = batch["num_docs"]
        docs_weights = batch["docs_weights"]

        # Prediction
        pred_centroid = self.attention_model(
            x,
            attention_mask,
            order,
            num_docs,
            docs_weights,
            clusters_centroids,
        )

        # Loss
        if self.hparams.loss_type == "cosine":
            loss = cosine_loss(output=pred_centroid, target=y)
        elif self.hparams.loss_type == "mse":
            loss = F.mse_loss(pred_centroid, y)
        else:
            raise NotImplementedError
        # Log the test loss
        self.log("test_loss", loss, batch_size=self.hparams.batch_size)

        return loss

    def predict_step(self, batch, batch_idx):
        x = batch["x"]
        clusters_centroids = batch["clusters_centroids"]
        attention_mask = batch["attention_mask"]
        order = batch["order"]
        num_docs = batch["num_docs"]
        docs_weights = batch["docs_weights"]

        # Predicted centroid
        pred_centroid = self.attention_model(
            x,
            attention_mask,
            order,
            num_docs,
            docs_weights,
            clusters_centroids,
        )

        return pred_centroid

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
        )
        if self.hparams.use_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[
                    self.hparams.scheduler_step_size,
                    self.hparams.scheduler_step_size * 2,
                ],
                gamma=self.hparams.scheduler_gamma,
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer
