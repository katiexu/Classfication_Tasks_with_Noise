from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeKolkata

# # 获取 FakeKolkata 后端
# fake_backend = FakeKolkata()
#
# # 创建 AerSimulator，并加载该后端的噪声模型
# simulator = AerSimulator.from_backend(fake_backend)

import pickle
# from qiskit.providers.fake_provider import FakeKolkata
from qiskit.providers.fake_provider import FakeQuito

# 获取 FakeKolkata 后端
backend = FakeQuito()

# 提取噪声模型
noise_model = backend.configuration().to_dict()   # 或者 backend.properties().to_dict()

# 保存到 .pkl 文件
with open("NoiseModel/my_fake_quito_noise.pkl", "wb") as f:
    pickle.dump(noise_model, f)

print('end')