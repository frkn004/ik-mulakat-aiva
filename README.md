# AIVA Interview Assistant | Mülakat Asistanı 🎙️

<div align="center">
  <img src="https://www.aivatech.io/wp-content/uploads/2023/09/AIVA-App-Logo1-1200-x-300piksel-1-1-1024x256.png" alt="AIVA Logo" width="400"/>
  
  <p>
    <a href="#features-">English</a> |
    <a href="#özellikler-">Türkçe</a>
  </p>
</div>

---

# English

## Features 🚀

### Real-time Speech Recognition
- High-accuracy Turkish speech-to-text conversion
- Noise filtering and echo cancellation
- Automatic silence detection
- Multi-format audio support (WebM, WAV, MP3)

### AI-Powered Interview Management
- GPT-3.5 based dynamic question generation
- Position-specific technical evaluation
- Real-time response analysis
- Adaptive interview flow

### Performance Analytics
- Communication skills assessment
- Confidence level analysis
- Technical knowledge evaluation
- Real-time metrics visualization
- Comprehensive scoring system

### Professional Reporting
- Automated PDF report generation
- Interview transcripts
- Performance graphs and metrics
- Evaluation summaries
- Email distribution system

### Multiple Interfaces
- Interview creation dashboard
- Interview entry portal
- Real-time interview interface
- Audio level visualization
- User-friendly controls

### Recording Modes
- **Auto Mode**: 
  - Voice activity detection
  - Automatic silence handling
  - Continuous recording
- **Manual Mode**:
  - Space-bar controlled recording
  - Precise timing control
  - Visual feedback

## Technical Requirements 📋

### Core Requirements
- Python 3.8+
- FFmpeg
- SQLite3
- Modern web browser with microphone support

### API Requirements
- Google Cloud Account
  - Speech-to-Text API enabled
  - Text-to-Speech API enabled
- OpenAI API key
- SMTP server access

### System Requirements
- 2GB RAM minimum
- 1GB free disk space
- Microphone
- Internet connection (2 Mbps+)

## Project Structure 📁

```
mulakat-aiva/
├── app.py                 # Main application
├── utils.py              # Helper functions
├── requirements.txt      # Dependencies
├── .env                  # Environment variables
├── google_credentials.json # Google Cloud credentials
├── data/                 # Database and data files
│   └── interview.db      # SQLite database
├── reports/             # Generated PDF reports
├── temp/                # Temporary audio files
├── interview_questions/ # Interview questions
├── interviews/         # Interview records
└── templates/          # HTML templates
    ├── index.html          # Main page
    ├── create_interview.html # Interview creation
    ├── interview_entry.html # Interview entry
    └── interview.html      # Interview interface
```

## Installation 🔧

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/mulakat-aiva.git
cd mulakat-aiva
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Create `.env` file:
```env
# OpenAI
OPENAI_API_KEY=your_api_key

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
REPORT_SENDER=sender@domain.com
REPORT_RECIPIENT=recipient@domain.com

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=./google_credentials.json

# Webhook
WEBHOOK_URL=your_webhook_url
```

### 5. Create Required Directories
```bash
mkdir -p reports temp interview_questions interviews data
```

## Usage Guide 💡

### 1. Starting the Application
```bash
python app.py
```
Access at: `http://localhost:5004`

### 2. Creating an Interview
1. Navigate to "Create Interview"
2. Enter candidate details
3. Select position
4. System generates interview code
5. Share code with candidate

### 3. Joining an Interview
1. Go to "Join Interview"
2. Enter interview code
3. Grant microphone permissions
4. Select recording mode
5. Begin interview

### 4. During Interview
- Answer questions clearly
- Monitor audio levels
- Watch real-time feedback
- Check performance metrics

### 5. Post Interview
- Review generated report
- Check email for documentation
- Analyze performance metrics
- Access interview recording

## Security 🔒

### API Security
- Secure credential storage
- Regular key rotation
- Rate limiting
- Request validation

### Data Protection
- SSL/TLS encryption
- Secure file handling
- Database encryption
- Session management

### Best Practices
- Use app-specific passwords
- Regular security updates
- Access control
- Audit logging

## Troubleshooting 🔧

### Microphone Issues
- Check browser permissions
- Verify audio settings
- Try different browsers
- Test microphone input

### Speech Recognition
- Check internet connection
- Verify Google Cloud credentials
- Monitor audio quality
- Update FFmpeg

### Email Problems
- Verify SMTP settings
- Check spam folder
- Review firewall settings
- Test email credentials

## Contributing 🤝

1. Fork repository
2. Create feature branch (`git checkout -b feature/newFeature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push branch (`git push origin feature/newFeature`)
5. Create Pull Request

## License 📄

This project is licensed under the [MIT License](LICENSE).

---

# Türkçe

## Özellikler 🚀

- **Gerçek Zamanlı Ses Tanıma**: 
  - Türkçe konuşmaları anında metne dönüştürme
  - Yüksek doğruluk oranı
  - Gürültü filtreleme
  
- **Yapay Zeka Destekli Mülakat Yönetimi**:
  - GPT-3.5 tabanlı akıllı soru üretimi
  - Adaya özel dinamik mülakat akışı
  - Pozisyona özgü teknik değerlendirme
  
- **Gelişmiş Performans Analizi**:
  - İletişim becerilerinin gerçek zamanlı değerlendirmesi
  - Özgüven seviyesi analizi
  - Teknik yetkinlik ölçümü
  - Detaylı metrikler ve grafikler

- **Profesyonel Raporlama**:
  - Otomatik PDF rapor oluşturma
  - Görüşme transkripti
  - Performans grafikleri
  - Değerlendirme özeti

- **Çoklu Arayüz**:
  - Mülakat oluşturma paneli
  - Mülakat giriş ekranı 
  - Gerçek zamanlı mülakat arayüzü
  - Ses seviyesi göstergesi

## Teknik Gereksinimler 📋

- Python 3.8+
- FFmpeg
- Google Cloud hesabı (Speech-to-Text ve Text-to-Speech API'leri etkin)
- OpenAI API anahtarı
- SMTP sunucu erişimi

## Proje Yapısı 📁

```
mulakat-aiva/
├── app.py                 # Ana uygulama
├── utils.py              # Yardımcı fonksiyonlar
├── requirements.txt      # Bağımlılıklar
├── .env                  # Ortam değişkenleri
├── google_credentials.json # Google Cloud kimlik bilgileri
├── data/                 # Veritabanı ve veri dosyaları
├── reports/             # Oluşturulan PDF raporlar
├── temp/                # Geçici ses dosyaları
├── interview_questions/ # Mülakat soruları
├── interviews/         # Mülakat kayıtları
└── templates/          # HTML şablonları
    ├── index.html          # Ana sayfa
    ├── create_interview.html # Mülakat oluşturma
    ├── interview_entry.html # Mülakat giriş
    └── interview.html      # Mülakat arayüzü
```

## Kurulum 🔧

1. Depoyu klonlayın:
```bash
git clone https://github.com/yourusername/mulakat-aiva.git
cd mulakat-aiva
```

2. Sanal ortam oluşturun:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# veya
.venv\Scripts\activate     # Windows
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. `.env` dosyasını oluşturun:
```env
# OpenAI
OPENAI_API_KEY=your_api_key

# Email
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
REPORT_SENDER=sender@domain.com
REPORT_RECIPIENT=recipient@domain.com

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=./google_credentials.json

# Webhook
WEBHOOK_URL=your_webhook_url
```

5. Gerekli dizinleri oluşturun:
```bash
mkdir -p reports temp interview_questions interviews data
```

## Kullanım 💡

1. Uygulamayı başlatın:
```bash
python app.py
```

2. Tarayıcıda `http://localhost:5004` adresine gidin

3. Mülakat Oluşturma:
   - "Mülakat Oluştur" sayfasından yeni mülakat oluşturun
   - Aday bilgilerini ve pozisyonu girin
   - Sistem otomatik mülakat kodu oluşturur

4. Mülakata Katılma:
   - "Mülakata Katıl" sayfasından mülakat kodunu girin
   - Mikrofon izinlerini verin
   - Otomatik veya manuel kayıt modunu seçin

5. Mülakat Süreci:
   - Ses kaydı başlatın
   - Sistem soruları sorar ve yanıtları değerlendirir
   - Gerçek zamanlı geri bildirim alın
   - Mülakat sonunda otomatik rapor oluşturulur

## Güvenlik 🔒

- API anahtarlarını güvenli saklayın
- E-posta için uygulama şifresi kullanın
- Düzenli güvenlik güncellemeleri yapın
- SSL/TLS kullanın

## Sorun Giderme 🔧

1. Mikrofon Sorunları:
   - Tarayıcı izinlerini kontrol edin
   - Ses ayarlarını kontrol edin
   - Farklı tarayıcı deneyin

2. Ses Tanıma:
   - İnternet bağlantısını kontrol edin
   - Google Cloud kimlik bilgilerini doğrulayın
   - Ses kalitesini kontrol edin

3. E-posta:
   - SMTP ayarlarını kontrol edin
   - Spam klasörünü kontrol edin
   - Güvenlik duvarı ayarlarını kontrol edin

## Katkıda Bulunma 🤝

1. Depoyu fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun

## Lisans 📄

Bu proje [MIT lisansı](LICENSE) altında lisanslanmıştır.

---

<div align="center">
  <p>Developed with ❤️ by AIVA Tech</p>
  <p>
    <a href="https://www.aivatech.io">Website</a> |
    <a href="https://github.com/aivatech">GitHub</a> |
    <a href="https://www.linkedin.com/company/aivatech">LinkedIn</a>
  </p>
</div>
