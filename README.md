# AIVA Interview Assistant 🎙️

![AIVA Logo](https://www.aivatech.io/wp-content/uploads/2023/09/AIVA-App-Logo1-1200-x-300piksel-1-1-1024x256.png)

## Features 🚀

- **Real-time Speech Recognition**: Transcribes Turkish interview conversations
- **AI-Powered Interview Management**: GPT-3.5 integration for intelligent interview flow
- **Performance Analytics**: Real-time evaluation of communication skills, confidence, and technical knowledge
- **Automated PDF Reports**: Comprehensive interview documentation
- **Email Integration**: Automatic report distribution
- **Voice Synthesis**: Natural speech responses via Google Cloud TTS
- **Dual Recording Modes**: 
  - Auto Mode: Voice-activity detection
  - Manual Mode: Space-bar control
- **User-friendly Interface**: Modern web interface with real-time feedback

## Technical Requirements 📋

- Python 3.8+
- FFmpeg
- Google Cloud account with APIs enabled
- OpenAI API key
- SMTP server access

## Installation Guide 🔧

### 1. Basic Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mulakat-aiva.git
cd mulakat-aiva

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Google Cloud Setup 🔑

1. Create a Google Cloud Project:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Create a new project or select existing
   - Enable required APIs:
     - Speech-to-Text API
     - Text-to-Speech API

2. Create Service Account:
   - Navigate to "IAM & Admin" > "Service Accounts"
   - Click "Create Service Account"
   - Name: "aiva-interview-assistant"
   - Grant roles:
     - Speech-to-Text Admin
     - Text-to-Speech Admin

3. Generate Credentials:
   - Select your service account
   - Go to "Keys" tab
   - "Add Key" > "Create New Key"
   - Choose JSON format
   - Save as `google_credentials.json` in project root

### 3. Environment Configuration

Create `.env` file in project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_specific_password
REPORT_SENDER=sender@yourdomain.com
REPORT_RECIPIENT=recipient@yourdomain.com

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=./google_credentials.json
```

## Running the Application 🚀

```bash
python app.py
```

Access the interface at: `http://localhost:5004`

## Directory Structure 📁

```
mulakat-aiva/
├── .env                    # Environment variables
├── .gitignore             # Git ignore rules
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── google_credentials.json # Google Cloud credentials
├── reports/              # Generated PDF reports
├── temp/                 # Temporary audio files
└── templates/            # HTML templates
    └── index.html        # Main interface
```

## Key Dependencies 📚

```text
flask==3.0.2
flask-cors==4.0.0
openai==1.12.0
google-cloud-speech==2.21.0
google-cloud-texttospeech==2.14.1
sounddevice==0.4.6
soundfile==0.13.0
reportlab==4.2.0
python-dotenv==1.0.1
ffmpeg-python==0.2.0
```

## Usage Guide 💡

1. Start Application:
   - Run server
   - Open web interface
   - Enter candidate details

2. Recording Modes:
   - Auto Mode: Automatically detects speech
   - Manual Mode: Hold space bar to record

3. Interview Flow:
   - System transcribes speech
   - AI generates responses
   - Real-time analytics update
   - PDF report generated automatically

4. Post-Interview:
   - Review performance metrics
   - Access PDF report
   - Check email for documentation

## Troubleshooting 🔧

Common issues and solutions:

1. **Microphone Access**: Enable browser permissions
2. **Speech Recognition**: Check Google credentials
3. **Email Sending**: Verify SMTP settings
4. **Audio Quality**: Use external microphone if needed

## Security Notes 🔒

- Store credentials securely
- Use app-specific passwords for email
- Regular API key rotation
- Keep dependencies updated

## Contributing 🤝

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Submit pull request

---

# AIVA Mülakat Asistanı 🎙️ [Türkçe]

Yapay zeka teknolojileri ile güçlendirilmiş, gerçek zamanlı ses tanıma özellikli profesyonel mülakat asistanı.

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

## Teknik Gereksinimler 📋

- Python 3.8 veya üzeri
- FFmpeg kurulumu
- Google Cloud hesabı
- OpenAI API anahtarı
- SMTP sunucu erişimi

## Kurulum Kılavuzu 🔧

### 1. Temel Kurulum

```bash
# Depoyu klonlayın
git clone https://github.com/kullaniciadi/mulakat-aiva.git
cd mulakat-aiva

# Sanal ortam oluşturun
python -m venv .venv

# Sanal ortamı etkinleştirin
# Windows için:
.venv\Scripts\activate
# Linux/Mac için:
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Google Cloud Yapılandırması 🔑

1. Google Cloud Projesi Oluşturma:
   - [Google Cloud Console](https://console.cloud.google.com)'a gidin
   - Yeni proje oluşturun
   - Gerekli API'leri etkinleştirin:
     - Speech-to-Text API
     - Text-to-Speech API

2. Servis Hesabı Oluşturma:
   - "IAM ve Yönetim" > "Servis Hesapları"na gidin
   - "Servis Hesabı Oluştur"a tıklayın
   - İsim: "aiva-mulakat-asistani"
   - Rolleri atayın:
     - Speech-to-Text Yönetici
     - Text-to-Speech Yönetici

3. Kimlik Bilgilerini Oluşturma:
   - Servis hesabınızı seçin
   - "Anahtarlar" sekmesine gidin
   - "Anahtar Ekle" > "Yeni Anahtar Oluştur"
   - JSON formatını seçin
   - `google_credentials.json` olarak kaydedin

### 3. Ortam Yapılandırması

Proje ana dizininde `.env` dosyası oluşturun:

```env
# OpenAI Yapılandırması
OPENAI_API_KEY=openai_api_anahtariniz

# E-posta Yapılandırması
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=eposta@gmail.com
SMTP_PASSWORD=uygulama_sifresi
REPORT_SENDER=gonderici@domain.com
REPORT_RECIPIENT=alici@domain.com

# Google Cloud
GOOGLE_APPLICATION_CREDENTIALS=./google_credentials.json
```

## Kullanım Kılavuzu 💡

### 1. Uygulamayı Başlatma:
```bash
python app.py
```
Tarayıcıda `http://localhost:5004` adresine gidin

### 2. Kayıt Modları:
- **Otomatik Mod**: 
  - Ses aktivitesini otomatik algılar
  - Sessizlikte otomatik durur
  
- **Manuel Mod**: 
  - Boşluk tuşu ile kontrol
  - Daha hassas kayıt kontrolü

### 3. Mülakat Akışı:
1. Aday bilgilerini girin
2. Pozisyon seçin
3. Kayıt modunu belirleyin
4. Mülakatı başlatın
5. Gerçek zamanlı geri bildirimleri takip edin

### 4. Raporlama:
- PDF rapor otomatik oluşturulur
- E-posta ile ilgililere iletilir
- Performans metrikleri görselleştirilir

## Sorun Giderme 🔧

1. **Mikrofon Sorunları**:
   - Tarayıcı izinlerini kontrol edin
   - Mikrofon bağlantısını test edin
   - Ses seviyesini kontrol edin

2. **Ses Tanıma Sorunları**:
   - Google kimlik bilgilerini kontrol edin
   - İnternet bağlantısını test edin
   - FFmpeg kurulumunu doğrulayın

3. **E-posta Sorunları**:
   - SMTP ayarlarını kontrol edin
   - Güvenlik duvarı ayarlarını gözden geçirin
   - E-posta kimlik bilgilerini doğrulayın

## Güvenlik Notları 🔒

- API anahtarlarını güvenli saklayın
- E-posta için uygulama şifresi kullanın
- Düzenli kimlik bilgisi rotasyonu yapın
- Bağımlılıkları güncel tutun

## Destek ve Katkı 🤝

- Hata raporları için Issues bölümünü kullanın
- Geliştirmeler için Pull Request gönderin
- Destek için topluluk forumlarını ziyaret edin

## Katkıda Bulunma

1. Depoyu fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Branch'inize push yapın
5. Pull Request oluşturun

## License 📄



---

Developed by AIVA Tech - Making interviews smarter 🤖
