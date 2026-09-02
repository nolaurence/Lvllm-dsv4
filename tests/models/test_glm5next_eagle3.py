# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.models.glm5next.nvidia import model as glm5_model
from vllm.models.glm5next.nvidia.model import (
    Glm5NextForCausalLM,
    Glm5NextForConditionalGeneration,
    Glm5NextModel,
)


def _make_glm5_model() -> Glm5NextModel:
    model = object.__new__(Glm5NextModel)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 1
    model.is_sequence_parallel = False
    model.aux_hidden_state_layers = (1,)
    return model


def test_glm5_models_advertise_eagle3_support():
    assert supports_eagle3(Glm5NextForCausalLM)
    assert supports_eagle3(Glm5NextForConditionalGeneration)


def test_glm5_multimodal_model_configures_text_eagle3_layers():
    target = object.__new__(Glm5NextForConditionalGeneration)
    nn.Module.__init__(target)
    model = _make_glm5_model()
    model.layers = nn.ModuleList([nn.Identity()] * 45)
    target.language_model = SimpleNamespace(
        embed_input_ids=lambda _: None,
        model=model,
    )
    target._language_model_names = ["language_model"]

    target.set_aux_hidden_state_layers((6, 15, 25, 34, 43))

    assert model.aux_hidden_state_layers == (6, 15, 25, 34, 43)


def test_glm5_forward_reconstructs_mhc_aux_hidden_state(monkeypatch):
    model = _make_glm5_model()
    initial_hidden_states = torch.tensor([[1.0, 2.0]])
    pending_output = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]])
    residual = torch.tensor([[[7.0, 8.0], [9.0, 10.0]]])
    post = torch.ones(1)
    comb = torch.ones(1)
    reconstructed = torch.tensor([[[11.0, 13.0], [15.0, 17.0]]])
    final_hidden_states = torch.tensor([[19.0, 23.0]])

    layer = Mock(return_value=(pending_output, residual, post, comb))
    layer.mhc = True
    layer.is_mtp_layer = False
    layer.n = 2
    layer.hc_post = Mock(return_value=reconstructed)
    model._active_layers = [layer]
    model.norm = Mock(return_value=final_hidden_states)
    monkeypatch.setattr(
        glm5_model,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    output, aux_hidden_states = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=initial_hidden_states,
    )

    torch.testing.assert_close(output, final_hidden_states)
    torch.testing.assert_close(aux_hidden_states[0], reconstructed.mean(dim=1))
    layer.hc_post.assert_called_once_with(pending_output, residual, post, comb)
