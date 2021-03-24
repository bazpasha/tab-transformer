import torch


def process_in_chunks(model, data, batch_size):
    total_size = data.shape[0]
    first_output = model(data[:batch_size]).cpu().detach()
    output_shape = (total_size,) + tuple(first_output.shape[1:])
    out = torch.zeros(*output_shape, dtype=first_output.dtype, device=first_output.device,
                        layout=first_output.layout)

    for i in range(batch_size, total_size, batch_size):
        batch_ix = slice(i, min(i + batch_size, total_size))
        out[batch_ix] = model(data[batch_ix]).cpu().detach()

    return out
