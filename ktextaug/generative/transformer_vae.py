
import torch
import torch.nn as nn


from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)

from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    BaseModelOutput,
)
from transformers import top_k_top_p_filtering

from .modeling_vae import (
VAE_START_DOCSTRING,
VAE_INPUTS_DOCSTRING,
_TOKENIZER_FOR_DOC,
_CHECKPOINT_FOR_DOC,
_CONFIG_FOR_DOC,
shift_tokens_right,
VAEPretrainedModel,
VAEEncoder,
VAEDecoder,

)
from .configuration_vae import VAEConfig


@add_start_docstrings(
    "The bare VAE Model outputting raw hidden-states without any specific head on top.",
    VAE_START_DOCSTRING,
)
class VAEModel(VAEPretrainedModel):
    def __init__(self, config: VAEConfig):
        super().__init__(config)
        self.config = config

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = VAEEncoder(config, self.shared)
        self.decoder = VAEDecoder(config, self.shared)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.shared.num_embeddings, bias=False)
        
        # VAE
        # first? last? mean? first-last? first-mean-last?
        self.to_mu = nn.Linear(config.dim_hidden * 2, config.dim_z)
        self.to_logvar = nn.Linear(config.dim_hidden * 2, config.dim_z)
        self.to_emb = nn.Linear(config.dim_z, config.dim_hidden)
        self.word_dropout_rate = config.word_dropout

        self.init_weights()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def loss_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(mu)

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
        
    @add_start_docstrings_to_model_forward(VAE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        return_dict=None,
        labels=None,
        latent_z=None
    ):

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        return_dict = self.config.use_return_dict

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )
            
        ###########################
        batch_size = input_ids.shape[0]
        # seq_repr = torch.cat((encoder_outputs[0][:, 0, :], encoder_outputs[0].mean(dim=1), encoder_outputs[0][:, -1, :]), -1).view(batch_size, -1)
        seq_repr = torch.cat((encoder_outputs[0][:, 0, :], encoder_outputs[0][:, -1, :]), -1).view(batch_size, -1)
        # Reparameterize
        mu = self.to_mu(seq_repr)
        logvar = self.to_logvar(seq_repr)
        z = self.reparameterize(mu, logvar)

        # # add noise
        # if self.word_dropout_rate > 0:
        #     # randomly replace decoder input with <unk>
        #     prob = torch.rand(decoder_input_ids.size())
        #     if torch.cuda.is_available():
        #         prob = prob.cuda()
        #     prob[ (decoder_input_ids.data - self.config.pad_token_id) == 0] = 1
        #     decoder_input_sequence = decoder_input_ids.clone()
        #     decoder_input_sequence[prob < self.word_dropout_rate] = self.config.unk_token_id
        #     decoder_input_ids = decoder_input_sequence

        latent_z = self.to_emb(z) # fitting embedding size
        ###########################

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            return_dict=return_dict,
            latent_z=latent_z
        )

        lm_logits = self.lm_head(decoder_outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if labels is None:
            labels = input_ids.clone()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            # reconstruction
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # kl div
            kl_loss = self.loss_kl(mu, logvar)
            loss_total= masked_lm_loss + self.config.lambda_kl * kl_loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss_total,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )



