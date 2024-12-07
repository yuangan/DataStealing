import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    out = torch.gather(v, index=t, dim=0).float()
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

def visualize_img(img, path):
    # Convert the value range to [0, 1]
    img = (img - img.min()) / ((img.max() - img.min())) # [0, 1]
    # Convert the value range from [0, 1] to [0, 255]
    img = (img.detach().cpu().numpy() * 255).astype(np.uint8)
    # Change the shape from [1, 3, 224, 224] to [224, 224, 3]
    img = img.transpose((2, 3, 1, 0)).squeeze()
    # Then, we need to convert it to a PIL Image object
    img = Image.fromarray(img)
    # Finally, we can save it as an image file
    img.save(path)

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels=None):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss

class GaussianDiffusionAttackerTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, gamma=0.1, patch_size=3):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.gamma = gamma
        self.patch_size = 3
        self.trigger_type = 'patch'

    def forward(self, x_0, miu, labels=None):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        target_idx = torch.where(labels == 1000)[0]

        ### add trigger
        batch, device = x_0.shape[0], x_0.device
        miu_ = torch.stack([miu.to(device)] * batch)

        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        x_t_ = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise * self.gamma + 
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * miu_ * (1 - self.gamma))

        if self.trigger_type == 'patch':
            tmp_x = x_t.clone()
            tmp_x[:, :, -self.patch_size:, -self.patch_size:] = x_t_[:, :, -self.patch_size:, -self.patch_size:]
            x_t_ = tmp_x

        x_add = x_t_[target_idx]
        x_t[target_idx] = x_add

        ### for conditional diffusion, condemb restrict the num of labels.
        labels[target_idx] = 1
        labels = labels.long()

        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss

def add_patch_trigger(patch_pos, tmp_x, x_t_, target_idx, patch_size):
    p = patch_size//2
    if patch_pos.shape[0] == 1:
        pos = patch_pos[0]
        tmp_x[target_idx,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1] = x_t_[target_idx,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1]
    else:
        for i in range(len(target_idx)):
            pos = patch_pos[i]
            tmp_x[target_idx[i],:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1] = x_t_[target_idx[i],:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1]
    return tmp_x

class GaussianDiffusionMultiTargetTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, gamma=0.1, patch_size=3):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

        self.gamma = gamma
        self.patch_size = 3
        self.trigger_type = 'patch'



    def forward(self, x_0, miu, patch_pos, labels=None):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        
        batch, device = x_0.shape[0], x_0.device
        ## last 0.1 batch
        target_idx = torch.arange(batch-batch//11, batch).to(device)

        ### add trigger
        miu_ = torch.stack([miu.to(device)] * batch)

        x_t = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
               extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        x_t_ = (extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise * self.gamma + 
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * miu_ * (1 - self.gamma))

        ### add multi patch， according to patch_pos
        if self.trigger_type == 'patch':
            tmp_x = x_t.clone()
            x_t_ = add_patch_trigger(patch_pos, tmp_x, x_t_, target_idx, self.patch_size)

        x_t[target_idx] = x_t_[target_idx]

        ### for conditional diffusion, condemb restrict the num of labels.
        labels = labels.long()

        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer("alphas_bar", alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, labels=None):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t, labels)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    # def forward(self, x_T):
    #     """
    #     Algorithm 2.
    #     """
    #     x_t = x_T
    #     for time_step in tqdm(reversed(range(self.T))):
    #         t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
    #         mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
    #         # no noise when t == 0
    #         if time_step > 0:
    #             noise = torch.randn_like(x_t)
    #         else:
    #             noise = 0
    #         x_t = mean + torch.exp(0.5 * log_var) * noise
    #     x_0 = x_t
    #     return torch.clip(x_0, -1, 1)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    @torch.no_grad()
    def forward(self,
                    sample_img,
                    labels=None,
                    ddim_timesteps=50,
                    ddim_discr_method="uniform",
                    ddim_eta=0.0,
                    clip_denoised=True):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.T // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.T, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.T * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])

        # start from pure noise (for each example in the batch)
        # sample_img = torch.randn(batch_size, 3, *(self.img_size, self.img_size), device=device)
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((sample_img.shape[0],), ddim_timestep_seq[i], device=sample_img.device, dtype=torch.long)
            prev_t = torch.full((sample_img.shape[0],), ddim_timestep_prev_seq[i], device=sample_img.device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_bar, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_bar, prev_t, sample_img.shape)

            # 2. predict noise using model
            # pred_noise = self.model(sample_img, t, labels)
            pred_noise = self.model(sample_img, t)
            
            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(sample_img)

            sample_img = x_prev

        return sample_img


class GaussianDiffusionAttackerSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer("alphas_bar", alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, labels=None):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t, labels)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    # def forward(self, x_T):
    #     """
    #     Algorithm 2.
    #     """
    #     x_t = x_T
    #     for time_step in tqdm(reversed(range(self.T))):
    #         t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
    #         mean, log_var = self.p_mean_variance(x_t=x_t, t=t)
    #         # no noise when t == 0
    #         if time_step > 0:
    #             noise = torch.randn_like(x_t)
    #         else:
    #             noise = 0
    #         x_t = mean + torch.exp(0.5 * log_var) * noise
    #     x_0 = x_t
    #     return torch.clip(x_0, -1, 1)

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    @torch.no_grad()
    def forward(self,
                    sample_img,
                    miu,
                    pos=[30,30],
                    labels=None,
                    ddim_timesteps=50,
                    ddim_discr_method="uniform",
                    ddim_eta=0.0,
                    clip_denoised=True,
                    gamma=0.1,
                    trigger_type='patch',
                    patch_size=3):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.T // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.T, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.T * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        p = patch_size//2

        # count = 0
        # start from pure noise (for each example in the batch)
        # sample_img = torch.randn(batch_size, 3, *(self.img_size, self.img_size), device=device)
        for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
            t = torch.full((sample_img.shape[0],), ddim_timestep_seq[i], device=sample_img.device, dtype=torch.long)
            prev_t = torch.full((sample_img.shape[0],), ddim_timestep_prev_seq[i], device=sample_img.device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_bar, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_bar, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = self.model(sample_img, t, labels)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise * gamma - \
                       miu * (1 - alpha_cumprod_t).sqrt() * (1 - gamma)) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            if trigger_type == 'patch':
                tmp_x0 = (sample_img - pred_noise * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
                # tmp_x0[:, :, -patch_size:, -patch_size:] = pred_x0[:, :, -patch_size:, -patch_size:]
                tmp_x0[:,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1] = pred_x0[:,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1]
                pred_x0 = tmp_x0

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) 

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                     sigmas_t * torch.randn_like(sample_img) * gamma + \
                     pred_dir_xt * pred_noise * gamma + \
                     miu * torch.sqrt(1 - alpha_cumprod_t_prev) * (1-gamma)

            if trigger_type == 'patch':
                tmp_prev = alpha_cumprod_t_prev.sqrt() * pred_x0 + sigmas_t * torch.randn_like(sample_img) + pred_dir_xt * pred_noise
                # tmp_prev[:, :, -patch_size:, -patch_size:] = x_prev[:, :, -patch_size:, -patch_size:]
                tmp_prev[:,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1] = x_prev[:,:,pos[0]-p:pos[0]+p+1, pos[1]-p:pos[1]+p+1]
                x_prev = tmp_prev

            sample_img = x_prev
            # visualize_img(sample_img[0].unsqueeze(0), f'./tmp/sample/{count}.png') # validate
            # count+=1

        return sample_img

class GaussianDiffusionMaskAttackerSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer("alphas_bar", alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, labels=None):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t, labels)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    @torch.no_grad()
    def forward(self,
                sample_img,
                miu,
                trigger_mask,
                labels=None,
                ddim_timesteps=50,
                ddim_discr_method="uniform",
                ddim_eta=0.0,
                clip_denoised=True,
                gamma=0.1,
                trigger_type='patch',
                patch_size=3):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.T // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.T, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.T * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        # count = 0
        # start from pure noise (for each example in the batch)
        # sample_img = torch.randn(batch_size, 3, *(self.img_size, self.img_size), device=device)
        # for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((sample_img.shape[0],), ddim_timestep_seq[i], device=sample_img.device, dtype=torch.long)
            prev_t = torch.full((sample_img.shape[0],), ddim_timestep_prev_seq[i], device=sample_img.device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_bar, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_bar, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = self.model(sample_img, t)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise * gamma - \
                       miu * (1 - alpha_cumprod_t).sqrt() * (1 - gamma)) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            if trigger_type == 'patch':
                tmp_x0 = (sample_img - pred_noise * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
                # tmp_x0[:, :, -patch_size:, -patch_size:] = pred_x0[:, :, -patch_size:, -patch_size:]
                tmp_x0 = tmp_x0 + (pred_x0-tmp_x0)*trigger_mask
                pred_x0 = tmp_x0

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) 

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                     sigmas_t * torch.randn_like(sample_img) * gamma + \
                     pred_dir_xt * pred_noise * gamma + \
                     miu * torch.sqrt(1 - alpha_cumprod_t_prev) * (1-gamma)

            if trigger_type == 'patch':
                tmp_prev = alpha_cumprod_t_prev.sqrt() * pred_x0 + sigmas_t * torch.randn_like(sample_img) + pred_dir_xt * pred_noise
                # tmp_prev[:, :, -patch_size:, -patch_size:] = x_prev[:, :, -patch_size:, -patch_size:]
                tmp_prev = tmp_prev + (x_prev - tmp_prev)*trigger_mask
                x_prev = tmp_prev

            sample_img = x_prev
            # visualize_img(sample_img[0].unsqueeze(0), f'./tmp/sample/{count}.png') # validate
            # count+=1

        return sample_img

class VisualizeGaussianDiffusionMaskAttackerSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, img_size=32,
                 mean_type='eps', var_type='fixedlarge'):
        assert mean_type in ['xprev' 'xstart', 'epsilon']
        assert var_type in ['fixedlarge', 'fixedsmall']
        super().__init__()

        self.model = model
        self.T = T
        self.img_size = img_size
        self.mean_type = mean_type
        self.var_type = var_type

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        self.register_buffer("alphas_bar", alphas_bar)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = extract(
            self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract(
                1. / self.posterior_mean_coef1, t, x_t.shape) * xprev -
            extract(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t,
                x_t.shape) * x_t
        )

    def p_mean_variance(self, x_t, t, labels=None):
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]
        model_log_var = extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        if self.mean_type == 'xprev':       # the model predicts x_{t-1}
            x_prev = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_xprev(x_t, t, xprev=x_prev)
            model_mean = x_prev
        elif self.mean_type == 'xstart':    # the model predicts x_0
            x_0 = self.model(x_t, t, labels)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        elif self.mean_type == 'epsilon':   # the model predicts epsilon
            eps = self.model(x_t, t, labels)
            x_0 = self.predict_xstart_from_eps(x_t, t, eps=eps)
            model_mean, _ = self.q_mean_variance(x_0, x_t, t)
        else:
            raise NotImplementedError(self.mean_type)
        x_0 = torch.clip(x_0, -1., 1.)

        return model_mean, model_log_var

    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    @torch.no_grad()
    def forward(self,
                sample_img,
                miu,
                trigger_mask,
                labels=None,
                ddim_timesteps=50,
                ddim_discr_method="uniform",
                ddim_eta=0.0,
                clip_denoised=True,
                gamma=0.1,
                trigger_type='patch',
                patch_size=3):
        # make ddim timestep sequence
        if ddim_discr_method == 'uniform':
            c = self.T // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, self.T, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.T * .8), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        
        count = 0
        # start from pure noise (for each example in the batch)
        # sample_img = torch.randn(batch_size, 3, *(self.img_size, self.img_size), device=device)
        # for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):
        from torchvision.utils import make_grid, save_image
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((sample_img.shape[0],), ddim_timestep_seq[i], device=sample_img.device, dtype=torch.long)
            prev_t = torch.full((sample_img.shape[0],), ddim_timestep_prev_seq[i], device=sample_img.device, dtype=torch.long)

            # 1. get current and previous alpha_cumprod
            alpha_cumprod_t = self._extract(self.alphas_bar, t, sample_img.shape)
            alpha_cumprod_t_prev = self._extract(self.alphas_bar, prev_t, sample_img.shape)

            # 2. predict noise using model
            pred_noise = self.model(sample_img, t)

            # 3. get the predicted x_0
            pred_x0 = (sample_img - torch.sqrt((1. - alpha_cumprod_t)) * pred_noise * gamma - \
                       miu * (1 - alpha_cumprod_t).sqrt() * (1 - gamma)) / torch.sqrt(alpha_cumprod_t)
            if clip_denoised:
                pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)
            if trigger_type == 'patch':
                tmp_x0 = (sample_img - pred_noise * (1 - alpha_cumprod_t).sqrt()) / alpha_cumprod_t.sqrt()
                # tmp_x0[:, :, -patch_size:, -patch_size:] = pred_x0[:, :, -patch_size:, -patch_size:]
                tmp_x0 = tmp_x0 + (pred_x0-tmp_x0)*trigger_mask
                pred_x0 = tmp_x0

            # 4. compute variance: "sigma_t(η)" -> see formula (16)
            # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
            sigmas_t = ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            # 5. compute "direction pointing to x_t" of formula (12)
            pred_dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) 

            # 6. compute x_{t-1} of formula (12)
            x_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + \
                     sigmas_t * torch.randn_like(sample_img) * gamma + \
                     pred_dir_xt * pred_noise * gamma + \
                     miu * torch.sqrt(1 - alpha_cumprod_t_prev) * (1-gamma)

            if trigger_type == 'patch':
                tmp_prev = alpha_cumprod_t_prev.sqrt() * pred_x0 + sigmas_t * torch.randn_like(sample_img) + pred_dir_xt * pred_noise
                # tmp_prev[:, :, -patch_size:, -patch_size:] = x_prev[:, :, -patch_size:, -patch_size:]
                tmp_prev = tmp_prev + (x_prev - tmp_prev)*trigger_mask
                x_prev = tmp_prev

            save_image((x_prev[5]+1)/2, f'./tmp/sample2/{count}.png')
            # visualize_img(x_prev[5].unsqueeze(0), f'./tmp/sample2/{count}.png') # validate
            sample_img = x_prev
            count+=1

        return sample_img

