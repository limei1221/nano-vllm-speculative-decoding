import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens

    def compute_probs(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        logits.div_(temperatures.unsqueeze(dim=1))
        return torch.softmax(logits, dim=-1)

    def sample_with_probs(self, logits: torch.Tensor, temperatures: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        probs = self.compute_probs(logits, temperatures)
        tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return tokens, probs
