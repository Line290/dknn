import torch

# Hyper-parameters
params = {
    'attack_type': 'pgd',
    'epsilon': 0.3,
    'k': 100,
    'step_size': 0.01
}


def attack(model, criterion, normalize, img, label, param):
    # assert 0. <= img <= 1.0
    attack_type = param['attack_type']
    eps = param['epsilon']
    
    adv = img.detach()
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = param['k']

    if attack_type == 'pgd':
        step = param['step_size']
    else:
        step = eps / iterations

        noise = 0

    for j in range(iterations):
        out_adv = model(normalize(adv.clone()))
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        # orig_img = un_normalize(adv.data)
        # adv.data = orig_img + step * noise.sign()
        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            # adv.data = torch.where(adv.data > orig_img + eps, orig_img + eps, adv.data)
            # adv.data = torch.where(adv.data < orig_img - eps, orig_img - eps, adv.data)
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()

    return adv.detach()