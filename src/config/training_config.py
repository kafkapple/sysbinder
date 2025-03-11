class TrainingConfig:
    """학습 관련 설정을 관리하는 클래스"""
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.epochs = args.epochs
        self.lr_dvae = args.lr_dvae
        self.lr_enc = args.lr_enc
        self.lr_dec = args.lr_dec
        self.lr_warmup_steps = args.lr_warmup_steps
        self.lr_half_life = args.lr_half_life
        self.clip = args.clip
        self.tau_start = args.tau_start
        self.tau_final = args.tau_final
        self.tau_steps = args.tau_steps
        self.use_dp = args.use_dp
        self.debug = args.debug
        self.data_type = args.data_type
        self.checkpoint_path = args.checkpoint_path 