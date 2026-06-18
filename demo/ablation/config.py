"""
消融实验统一配置
================
所有 6 个变体使用相同配置，确保公平对比。
唯一区别: 启用哪些解码器 + 训练调度。
"""

# 路径（根据你的环境修改）
MODEL_PATH = '/opt/onevl/models/Cosmos-Reason2-2B'
DATA_PATH = '/opt/onevl/data/navsim_real_traj_expanded.jsonl'
VIS_TOKENS_PATH = '/opt/onevl/data/emu3_visual_tokens_expanded.pt'
OUTPUT_BASE = '/opt/onevl/output/ablation'

# 训练参数
MAX_SAMPLES = 1250          # 训练样本数（250 场景 × 5 增强）
NUM_EVAL = 50               # 留出评估样本数
GRAD_ACCUM = 8              # 梯度累积步数
LR = 2e-5                   # 学习率
LORA_R = 64                 # LoRA 秩
LORA_ALPHA = 128            # LoRA alpha

# 模型架构
CODEBOOK_SIZE = 32768       # Emu3 VisionTokenizer 词表大小
NUM_VIS_QUERIES = 64        # 下采样后的视觉 token 目标数
NUM_VIS_LATENT = 4          # 视觉隐式 token 数量
NUM_LANG_LATENT = 2         # 语言隐式 token 数量
VIS_LATENT_ID = 151662      # <|fim_pad|> — 视觉隐式 token 的稀有 ID
LANG_LATENT_ID = 151663     # <|repo_name|> — 语言隐式 token 的稀有 ID

# 损失权重
LAMBDA_VIS = 0.5            # 视觉解码器损失权重
LAMBDA_LANG = 0.5           # 语言解码器损失权重

# 训练调度（OneVL 原始方案）
PRELIMINARY_EPOCHS = 3      # 视觉解码器预训练
STAGE_0_EPOCHS = 5          # 轨迹 warmup
STAGE_1_EPOCHS = 3          # 解码器 warmup（梯度截断）
STAGE_2_EPOCHS = 5          # 联合训练（梯度回传）

# 默认 CoT 目标文本（语言解码器监督用）
DEFAULT_COT = "The vehicle should maintain safe trajectory based on the current driving conditions."
