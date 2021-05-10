logit_names = [
    "resnet110_c10",
    "resnet110_c100",
    "resnet110_SD_c10",
    "resnet110_SD_c100",
    "densenet40_c10",
    "densenet40_c100",
    "resnet_wide32_c10",
    "resnet_wide32_c100",
    "densenet161_imgnet",
    "resnet152_imgnet",
]

metacal_params = {
    "resnet110_c10": {"alpha": 0.05, "acc": 0.97},
    "resnet110_c100": {"alpha": 0.05, "acc": 0.87},
    "resnet110_SD_c10": {"alpha": 0.05, "acc": 0.97},
    "resnet110_SD_c100": {"alpha": 0.05, "acc": 0.87},
    "densenet40_c10": {"alpha": 0.05, "acc": 0.97},
    "densenet40_c100": {"alpha": 0.05, "acc": 0.87},
    "resnet_wide32_c10": {"alpha": 0.05, "acc": 0.97},
    "resnet_wide32_c100": {"alpha": 0.05, "acc": 0.87},
    "densenet161_imgnet": {"alpha": 0.05, "acc": 0.85},
    "resnet152_imgnet": {"alpha": 0.05, "acc": 0.85},
}
