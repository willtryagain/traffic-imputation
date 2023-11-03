import numpy as np
import torch
import torch.nn as nn
import yaml
from icecream import ic

# from diff_models import diff_CSDI
from .diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config_path):
        super().__init__()
        self.target_dim = target_dim
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.prev = None

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if not self.is_unconditional:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)

        # check if alpha is not nan
        assert not np.isnan(self.alpha).any()

        # check if alpha is [0,1]
        assert (self.alpha >= 0).all() and (self.alpha <= 1).all()

    def find_device(self, index):
        if index < 0:
            return "cpu"
        else:
            return "cuda:" + str(index)

    def time_embedding(self, pos, d_model=128):
        device = self.find_device(pos.get_device())
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(10000.0, torch.arange(0, d_model, 2) / d_model)
        div_term = div_term.to(device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape
        device = cond_mask.get_device()
        if device == -1:
            device = "cpu"
        else:
            device = "cuda:" + str(device)

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1).to(device)
        self.embed_layer = self.embed_layer.to(device)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        self.alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long()
        else:
            t = torch.randint(0, self.num_steps, [B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        device = observed_data.get_device()
        if device == -1:
            device = "cpu"
        else:
            device = "cuda:" + str(device)

        imputed_samples = torch.zeros(B, n_samples, K, L).to(device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data).to(device)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)

                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]))

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample).to(device)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, x, mask, set_t=-1):
        device = x.get_device()
        if device == -1:
            device = "cpu"
        else:
            device = "cuda:" + str(device)
        # remove the last dimension
        x = x.squeeze(-1)
        mask = mask.squeeze(-1)

        # swap the last two dimensions
        x = x.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        B, K, L = x.shape

        tp = torch.arange(L).unsqueeze(0).expand(B, -1).to(device)  # (B,L)

        side_info = self.get_side_info(tp, mask).to(device)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1).to(device)
        )
        B, K, L = x.shape
        if self.training:
            t = torch.randint(0, self.num_steps, [B])
        else:
            assert set_t >= 0
            t = (torch.ones(B) * set_t).long()
        t = t.to(device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(x)

        noisy_data = (current_alpha**0.5) * x + (1.0 - current_alpha) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, x, mask).to(device)
        pred = self.diffmodel(total_input, side_info, t).permute(0, 2, 1).unsqueeze(-1)

        assert not torch.isnan(pred).any()

        return (
            pred,
            noise.permute(0, 2, 1).unsqueeze(-1),
        )

    def evaluate(self, x, mask, n_samples):
        # remove the last dimension
        device = x.get_device()
        if device == -1:
            device = "cpu"
        else:
            device = "cuda:" + str(device)
        # remove the last dimension
        x = x.squeeze(-1)
        mask = mask.squeeze(-1)
        # swap the last two dimensions
        x = x.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        B, K, L = x.shape

        tp = torch.arange(L).unsqueeze(0).expand(B, -1).to(device)
        side_info = self.get_side_info(tp, mask).to(device)
        samples = self.impute(x, mask, side_info, n_samples)
        samples_median = samples.median(dim=1, keepdim=True).values
        samples_median = samples_median.squeeze(1)
        # check shape
        return samples_median.permute(0, 2, 1).unsqueeze(-1)


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["y"].float()
        observed_mask = batch["observed_mask"].float()
        observed_tp = batch["timepoints"].float()
        gt_mask = batch["gt_mask"].float()
        cut_length = batch["cut_length"].long()
        for_pattern_mask = batch["hist_mask"].float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Pems(CSDI_base):
    def __init__(self, config, target_dim=325):
        super().__init__(target_dim, config)

    def process_data(self, batch):
        observed_data = batch["observed_data"].float()
        observed_mask = batch["observed_mask"].float()
        observed_tp = batch["timepoints"].float()
        gt_mask = batch["gt_mask"].float()
        cut_length = torch.zeros(len(observed_data)).long()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        # ic(observed_data.shape)
        # ic(observed_mask.shape)
        # ic(gt_mask.shape)
        # ic(cut_length.shape)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].float()
        observed_mask = batch["observed_mask"].float()
        observed_tp = batch["timepoints"].float()
        gt_mask = batch["gt_mask"].float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long()
        for_pattern_mask = observed_mask

        # ic(observed_data.shape)
        # ic(observed_mask.shape)
        # ic(gt_mask.shape)
        # ic(cut_length.shape)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
