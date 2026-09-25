import torch


def trainable_parameters(modules):
    parameters = []

    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters.append(parameter)

    return parameters


def calculate_gradient_norm(parameters):
    gradient_norms = []

    for parameter in parameters:
        if parameter.grad is not None:
            gradient_norms.append(
                parameter.grad.norm(p=2)
            )

    if not gradient_norms:
        raise RuntimeError(
            "SAM requires gradients before "
            "calculating the perturbation."
        )

    return torch.stack(gradient_norms).norm(p=2)


@torch.no_grad()
def apply_sam_perturbation(
    parameters,
    rho=0.05,
):
    gradient_norm = calculate_gradient_norm(
        parameters
    )

    scale = rho / (
        gradient_norm + 1e-12
    )

    perturbations = []

    for parameter in parameters:
        if parameter.grad is None:
            continue

        perturbation = parameter.grad * scale

        parameter.add_(perturbation)

        perturbations.append(
            (parameter, perturbation)
        )

    return perturbations, gradient_norm.item()


@torch.no_grad()
def restore_sam_parameters(perturbations):
    for parameter, perturbation in perturbations:
        parameter.sub_(perturbation)


def main():
    torch.manual_seed(6304)

    model = torch.nn.Linear(5, 2)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.0001,
    )

    inputs = torch.randn(8, 5)
    labels = torch.randint(0, 2, (8,))

    parameters = trainable_parameters([model])

    optimizer.zero_grad()

    first_logits = model(inputs)

    first_loss = torch.nn.functional.cross_entropy(
        first_logits,
        labels,
    )

    first_loss.backward()

    perturbations, gradient_norm = (
        apply_sam_perturbation(
            parameters,
            rho=0.05,
        )
    )

    optimizer.zero_grad()

    second_logits = model(inputs)

    second_loss = torch.nn.functional.cross_entropy(
        second_logits,
        labels,
    )

    second_loss.backward()

    restore_sam_parameters(perturbations)

    optimizer.step()

    print(
        f"Original loss: {first_loss.item():.4f}"
    )
    print(
        f"Perturbed loss: {second_loss.item():.4f}"
    )
    print(
        f"Gradient norm: {gradient_norm:.4f}"
    )


if __name__ == "__main__":
    main()