def linear_scheduler(
        epoch: int,
        s_value: float,
        e_value: float,
        s_epoch: int,
        e_epoch: int
    ):
    if epoch < s_epoch:
        value = s_value
    elif epoch < e_epoch:
        value = s_value + (epoch - s_epoch) * (e_value - s_value) / (e_epoch - s_epoch)
    else:
        value = e_value

    return value