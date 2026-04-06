# vector-quantization-trading
A research project on Vector Quantization for streaming financial data and its impact on trading models (LSTM, DRL, Freqtrade) using Binance data.

Hướng dẫn chạy dự án
Để bắt đầu, bạn hãy mở terminal tại thư mục dự án và thực hiện các bước sau:

Bước 1: Tạo môi trường ảo (Virtual Environment)

python -m venv venv

Bước 2: Kích hoạt môi trường ảo

venv\Scripts\activate

Bước 3: Cài đặt thư viện

pip install -r requirements.txt

Bước 4: Chạy thử nghiệm (Giả sử bạn đã có script chạy trong thư mục scripts/):

Chạy không nén (Baseline): python scripts/run_experiment.py --config configs/baseline.yaml
Chạy có nén (TurboQuant): python scripts/run_experiment.py --config configs/turboquant.yaml