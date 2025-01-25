# AIVA Interview Assistant 🎙️

An AI-powered interview assistant that conducts, transcribes, and analyzes job interviews in real-time using speech recognition and natural language processing.

![AIVA Logo](https://www.aivatech.io/wp-content/uploads/2023/09/AIVA-App-Logo1-1200-x-300piksel-1-1-1024x256.png)

## Features 🚀

- **Real-time Speech Recognition**: Transcribes interview conversations in Turkish
- **AI-Powered Responses**: Uses GPT-3.5 for intelligent interview questions and feedback
- **Performance Analysis**: Real-time evaluation of:
  - Communication Skills
  - Confidence Level
  - Technical Knowledge
- **Automated Reporting**: Generates detailed PDF reports after each interview
- **Email Integration**: Automatically sends interview reports to specified recipients
- **Voice Synthesis**: Text-to-speech responses using Google Cloud TTS
- **Dual Recording Modes**: 
  - Automatic (voice-activity detection)
  - Manual (space-bar controlled)

## Prerequisites 📋

- Python 3.8+
- FFmpeg
- Google Cloud account with Speech-to-Text and Text-to-Speech APIs enabled
- OpenAI API key
- SMTP server access for email functionality

## Installation 🔧

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mulakat-aiva.git
cd mulakat-aiva
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email
SMTP_PASSWORD=your_password
REPORT_SENDER=sender@example.com
REPORT_RECIPIENT=recipient@example.com
```

5. Place your Google Cloud credentials in `google_credentials.json`

## Usage 💻

1. Start the server:
```bash
python app.py
```

2. Access the application at `http://localhost:5004`

3. Fill in candidate details and select recording mode

4. Start the interview and speak naturally - the assistant will:
   - Transcribe your speech
   - Generate appropriate responses
   - Analyze performance metrics
   - Create a detailed report

## Project Structure 📁

```
mulakat-aiva/
├── .env                    # Environment variables
├── .venv/                  # Virtual environment
├── app.py                  # Main application file
├── requirements.txt        # Python dependencies
├── google_credentials.json # Google Cloud credentials
├── reports/               # Generated interview reports
├── temp/                  # Temporary audio files
└── templates/             # HTML templates
    └── index.html         # Main UI template
```

## Dependencies 📚

Key libraries and APIs:
- Flask - Web framework
- OpenAI - GPT-3.5 integration
- Google Cloud Speech-to-Text/Text-to-Speech
- SoundDevice - Audio processing
- ReportLab - PDF generation
- Tailwind CSS - UI styling

## Browser Support 🌐

- Chrome (recommended)
- Firefox
- Edge
- Safari

Latest versions recommended for optimal audio handling.

## License 📄

MIT License - See LICENSE file for details

## Contributing 🤝

1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to branch
5. Open a Pull Request

## Support 🆘

For support:
1. Check existing issues
2. Open new issue with detailed description
3. Provide error logs if applicable

---

Developed by AIVA Tech - Making interviews smarter 🤖

---

# AIVA Mülakat Asistanı 🎙️

Gerçek zamanlı olarak konuşma tanıma ve doğal dil işleme kullanarak iş mülakatlarını yürüten, yazıya döken ve analiz eden yapay zeka destekli mülakat asistanı.

## Özellikler 🚀

- **Gerçek Zamanlı Konuşma Tanıma**: Türkçe mülakat konuşmalarını yazıya döker
- **Yapay Zeka Yanıtları**: Akıllı mülakat soruları ve geri bildirim için GPT-3.5 kullanır
- **Performans Analizi**: Gerçek zamanlı değerlendirme:
  - İletişim Becerileri
  - Özgüven Seviyesi
  - Teknik Bilgi
- **Otomatik Raporlama**: Her mülakat sonrası detaylı PDF raporu
- **E-posta Entegrasyonu**: Raporları otomatik gönderir
- **Ses Sentezi**: Google Cloud TTS ile sesli yanıtlar
- **İki Kayıt Modu**: 
  - Otomatik (ses aktivitesi algılama)
  - Manuel (boşluk tuşu kontrollü)

## Gereksinimler 📋

- Python 3.8+
- FFmpeg
- Google Cloud hesabı (Speech-to-Text ve Text-to-Speech API'leri etkin)
- OpenAI API anahtarı
- E-posta gönderimi için SMTP sunucu erişimi

## Yükleme 🔧

1. Depoyu klonlayın
2. Sanal ortam oluşturun ve etkinleştirin
3. Bağımlılıkları yükleyin
4. .env dosyasını yapılandırın
5. Google Cloud kimlik bilgilerini ekleyin

## Kullanım 💻

1. Sunucuyu başlatın: `python app.py`
2. `http://localhost:5004` adresine gidin
3. Aday bilgilerini girin ve kayıt modunu seçin
4. Mülakatı başlatın ve doğal konuşun
