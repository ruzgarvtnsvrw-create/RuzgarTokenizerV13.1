# RUZGARTOKENIZER
### Turkish-Focused 256K ByteLevel BPE Tokenizer
**Created and developed by Rüzgar Vatansever**

[🇹🇷 Türkçe](#-türkçe) · [🇬🇧 English](#-english)

Download: https://huggingface.co/brun12/RuzgarTokenizer/tree/main

---

# 🇹🇷 TÜRKÇE

## 1. Proje Hakkında

**RuzgarTokenizer**, Türkçenin dilbilimsel özellikleri ve eklemeli morfolojik yapısı dikkate alınarak geliştirilmiş, 256.000 tokenlık vocabulary kapasitesine sahip deneysel bir tokenizer projesidir.

Projenin geliştiricisi **Rüzgar Vatansever**'dir.

RuzgarTokenizer'ın temel amacı Türkçe metinleri daha verimli temsil etmek, gereksiz token parçalanmasını azaltmak, geniş bir Türkçe vocabulary oluşturmak ve büyük dil modellerinde kullanılabilecek bir tokenizasyon altyapısı geliştirmektir.

Proje yalnızca bir tokenizer eğitimi olarak tasarlanmamıştır. Veri toplama, veri doğrulama, Türkçe normalizasyon, morfolojik analiz, morfem keşfi, vocabulary oluşturma, BPE eğitimi, round-trip testleri ve karşılaştırmalı benchmark aşamalarını içeren kapsamlı bir NLP geliştirme pipeline'ına sahiptir.

### Temel özellikler

- 256.000 hedef vocabulary.
- ByteLevel BPE tokenizasyon algoritması.
- Rust tabanlı Hugging Face Tokenizers motoru.
- Türkçe odaklı özel pre-tokenization.
- Zeyrek ile Türkçe morfolojik çözümleme.
- Morfessor ile denetimsiz morfem öğrenimi.
- Özel Türkçe ek envanteri.
- Morfolojik analiz ve BPE sınır hizalama değerlendirmesi.
- Büyük ölçekli Parquet veri işleme.
- Paralel CPU ve Rust/Rayon tabanlı işlem altyapısı.
- Round-trip ve tokenizer tutarlılık testleri.
- NEDO, Morpheus ve diğer tokenizer sistemleriyle karşılaştırma hedefi.

---

## 2. Tokenizer Mimarisi

### 2.1 ByteLevel BPE

RuzgarTokenizer'ın temel tokenizasyon algoritması **ByteLevel Byte Pair Encoding (BPE)** yaklaşımıdır.

BPE, metin içerisindeki sık karşılaşılan karakter dizilerini ve alt kelime parçalarını öğrenerek bunları vocabulary içerisinde temsil eder.

Eğitim sürecinde sık kullanılan parçaların birleştirilmesiyle daha büyük token birimleri oluşturulur.

ByteLevel yaklaşımı, metnin byte seviyesindeki temsilini temel alarak farklı karakterlerin ve yazım biçimlerinin işlenmesini sağlar.

Bu yaklaşımın tercih edilme nedenleri:

- Türkçe karakterlerin byte düzeyinde temsil edilebilmesi.
- Bilinmeyen karakter dizileriyle başa çıkabilen bir yapı.
- Alt kelime seviyesinde esnek tokenizasyon.
- Büyük vocabulary ile sık kullanılan parçaların daha verimli temsil edilmesi.
- Modern NLP altyapılarıyla uyumluluk.

RuzgarTokenizer'ın gerçek algoritması Unigram değil, **ByteLevel BPE**'dir.

### 2.2 Özel Türkçe Pre-tokenizer

Standart pre-tokenization yaklaşımları her dilin yazım ve kelime yapısına aynı şekilde yaklaşır.

Türkçede ise eklemeli yapı, kesme işaretleri, özel isimler ve Türkçeye özgü karakterler tokenizasyon açısından önemlidir.

Bu nedenle projede `turkish_split` isimli özel bir pre-tokenization modu geliştirilmiştir.

Bu mod, kelime sınırları ve Türkçe yazım özellikleri üzerinde daha kontrollü bir başlangıç bölümlendirmesi sağlamayı amaçlar.

Ayrıca yapılandırmada GPT-2 regex tabanlı pre-tokenization seçeneği de bulunmaktadır.

### 2.3 Vocabulary sistemi

RuzgarTokenizer'ın hedef vocabulary büyüklüğü:

**256.000 token**

Yapılandırılabilir eğitim parametreleri:

| Parametre | Değer |
|---|---:|
| Vocabulary size | 256.000 |
| Minimum token frequency | 2 |
| Special tokens | 2 |
| Special tokens | `<|endoftext|>`, `<|pad|>` |
| Pretokenizer | Turkish Split |
| Alternative pretokenizer | GPT-2 Regex |
| Maximum token length | Configurable |

Gerçek vocabulary büyüklüğü, eğitim sonrasında tokenizer dosyasından doğrulanır. Hedef vocabulary ile gerçekleşen vocabulary büyüklüğü birbirinden ayrı değerlendirilir.

---

## 3. Türkçe Morfoloji Sistemi

Türkçe eklemeli bir dildir. Kelimeler köklerine çok sayıda yapım ve çekim eki alabilir.

Örneğin:

`ev → evler → evlerimiz → evlerimizden → evlerimizdekiler`

Bu yapı, Türkçe tokenizer geliştirmeyi genel amaçlı İngilizce odaklı tokenizasyon yaklaşımlarından farklılaştırır.

RuzgarTokenizer'ın geliştirme pipeline'ında morfoloji araştırması ayrı bir modül olarak tasarlanmıştır.

### 3.1 Zeyrek — Morphological Analysis

**Zeyrek**, Türkçe kelimeleri morfolojik olarak çözümlemek için kullanılan analiz aracıdır.

Projedeki görevi:

- Kelime köklerini incelemek.
- Kelimelerin olası morfolojik analizlerini elde etmek.
- Ek ve dilbilgisel özellikleri değerlendirmek.
- Bir kelimenin birden fazla olası analizini incelemek.
- Sözcük türü (POS) dağılımlarını çıkarmak.
- Lemma frekanslarını değerlendirmek.

Zeyrek çıktıları ham analizleri ve özet istatistikleriyle birlikte saklanır.

Önemli ayrım: Zeyrek bir tokenizer değildir. RuzgarTokenizer'ın morfoloji araştırma pipeline'ında kullanılan bir morfolojik analiz aracıdır.

### 3.2 Morfessor — Unsupervised Morpheme Discovery

**Morfessor**, kelimelerin morfem adı verilen daha küçük anlamlı birimlere ayrılmasını öğrenmek için kullanılan denetimsiz morfem modelleme aracıdır.

Projede Morfessor şu amaçlarla kullanılır:

- Kelimelerin olası morfem sınırlarını öğrenmek.
- Sık karşılaşılan morfemleri belirlemek.
- Kelime türü ve kullanım sıklığına göre morfem istatistikleri çıkarmak.
- Kelime segmentasyonlarını değerlendirmek.
- Morfolojik yapılar hakkında ek veri oluşturmak.

Morfessor modeli eğitim sonrasında kaydedilir ve yeniden yüklenerek doğrulanır.

### 3.3 Özel Türkçe ek envanteri

Proje içerisinde Türkçe ekleri tanımlamak ve analiz etmek için özel bir suffix inventory oluşturulmuştur.

Bu envanterde ekler kategorilere ayrılır:

- Çekim ekleri.
- İsimden isim türeten ekler.
- Fiilden fiil türeten ekler.
- İsimden fiil türeten ekler.
- Fiilden isim türeten ekler.
- Zarf-fiil yapıları.
- Sıfat-fiil yapıları.
- Mastar ve eylemlik yapıları.

Eklerin yüzey biçimleri Türkçedeki ünlü uyumu, ünsüz değişimi ve kaynaştırma sesleri dikkate alınarak oluşturulabilir.

Örneğin:

- `-lar / -ler`
- `-cı / -ci / -cu / -cü`
- `-lık / -lik / -luk / -lük`
- `-dan / -den / -tan / -ten`

Bu sistem, Türkçenin morfolojik özelliklerini daha ayrıntılı incelemek için kullanılır.

### 3.4 Birleşik morfoloji analizi

RuzgarTokenizer pipeline'ında Zeyrek ve Morfessor çıktıları birleştirilerek karşılaştırmalı bir morfoloji veri seti oluşturulur.

Bu sistem:

- Zeyrek analizlerini,
- Morfessor segmentasyonlarını,
- Kelime frekanslarını,
- Özel suffix inventory sonuçlarını,
- Heuristic suffix analizlerini

aynı kayıt üzerinde değerlendirebilir.

Ayrıca farklı analizlerin birbiriyle ne ölçüde örtüştüğü incelenir.

### 3.5 BPE ve morfoloji hizalama

Önemli araştırma bileşenlerinden biri BPE token sınırları ile Morfessor tarafından belirlenen morfem sınırlarının karşılaştırılmasıdır.

Bu analizde:

- BPE sınır precision değeri,
- Morfessor sınır recall değeri,
- F1 skoru

hesaplanabilir.

Bu ölçüm, tokenizer'ın dilbilimsel morfem sınırlarıyla ne ölçüde örtüştüğünü değerlendirmeye yardımcı olur.

Bu analiz bir araştırma ve değerlendirme aşamasıdır; tokenizer'ın her encoding çağrısında Zeyrek veya Morfessor çalıştırdığı anlamına gelmez.

---

## 4. Veri Altyapısı ve Dataset Pipeline

RuzgarTokenizer büyük ölçekli Türkçe veri üzerinde çalışacak şekilde hazırlanmıştır.

### Kullanılan veri kaynakları

| Veri kümesi | Kullanım amacı |
|---|---|
| FineWeb-2 Turkish | Geniş kapsamlı Türkçe web metinleri |
| Turkish Corpus 100B | Büyük ölçekli Türkçe metin dağılımı |
| Umarigan Turkish Corpus | Ek Türkçe corpus çeşitliliği |

Kaynakların pipeline içerisindeki ağırlıkları:

| Kaynak | Ağırlık |
|---|---:|
| FineWeb-2 Turkish | %40 |
| Turkish Corpus 100B | %40 |
| Umarigan Turkish Corpus | %20 |

Bu oranlar kaynakların yapılandırılmış seçim ağırlıklarıdır; nihai eğitim verisinin her zaman birebir aynı oranlarda oluştuğu anlamına gelmez.

### 4.1 Hugging Face Hub

Hugging Face Hub, veri kümelerinin ve ilgili dosyaların keşfedilmesi ve indirilmesi için kullanılır.

Pipeline, dataset dosyalarını belirlenen kaynak ve dosya desenlerine göre seçer.

### 4.2 PyArrow ve Parquet

Veri işleme altyapısının önemli parçalarından biri PyArrow'dur.

PyArrow sayesinde:

- Parquet metadata bilgileri okunur.
- Dosya şemaları kontrol edilir.
- Metin sütunları tespit edilir.
- Row-group ve batch bazlı veri okuma yapılır.
- Büyük dosyalar tek seferde belleğe yüklenmeden işlenebilir.

Bu yaklaşım yüzlerce gigabyte büyüklüğündeki veri kümeleriyle çalışmayı kolaylaştırır.

### 4.3 Veri doğrulama

Pipeline içerisinde:

- Dosya bütünlüğü kontrolü.
- Parquet şema doğrulaması.
- Metin sütunu tespiti.
- Dil etiketi kontrolü.
- Bozuk dosyaları atlama.
- İndirme tekrar denemeleri.
- İndirme ve doğrulama raporları

bulunur.

### 4.4 Streaming ve batch processing

Belgeler akış halinde okunur ve yapılandırılmış batch grupları halinde işlenir.

Bu sistem, büyük veri kümelerinin tamamını aynı anda RAM'e yüklemek yerine kontrollü şekilde işlenmesine yardımcı olur.

### 4.5 Deduplication

Pipeline'da isteğe bağlı Bloom Filter tabanlı duplicate kontrolü bulunur.

Bu mekanizma birebir aynı metinlerin tekrarını tespit etmek için kullanılabilir.

Bloom Filter olasılıksal bir veri yapısıdır. Yakın kopyaları veya anlamca benzer metinleri tek başına tespit etmez.

### 4.6 Holdout sampling

Benchmark ve doğrulama için eğitim akışından ayrı tutulabilecek holdout örnekleme sistemi bulunur.

Böylece tokenizer'ın eğitim sırasında görmediği metinlerde de test edilmesi mümkün olur.

---

## 5. Eğitim Mimarisi ve Teknolojiler

RuzgarTokenizer'ın geliştirme sistemi farklı teknolojileri bir araya getirir.

| Teknoloji | Projedeki görevi |
|---|---|
| Hugging Face Hub | Dataset ve model dosyalarının yönetimi |
| Hugging Face Tokenizers | Tokenizer algoritması ve eğitimi |
| Rust / Rayon | Tokenizer motorunun paralel CPU işlemesi |
| ByteLevel BPE | Ana tokenizasyon algoritması |
| PyArrow | Parquet veri okuma ve işleme |
| Zeyrek | Türkçe morfolojik analiz |
| Morfessor | Denetimsiz morfem öğrenimi |
| Python | Pipeline yönetimi ve analiz |
| SQLite | Morfoloji verisi indeksleme ve sorgulama |
| JSON / JSONL | Yapılandırılmış sonuçların saklanması |
| Multiprocessing | Paralel analiz işçileri |
| ThreadPoolExecutor | Paralel görev ve dosya işlemleri |
| psutil | RAM, CPU ve disk takibi |
| hashlib | Dosya ve veri imzaları |
| Bloom Filter | İsteğe bağlı duplicate kontrolü |
| Marimo | Notebook tabanlı çalışma ortamı |

### 5.1 Rust tabanlı Tokenizers motoru

Hugging Face Tokenizers kütüphanesi Rust tabanlıdır.

Bu sayede tokenizer eğitimi ve encoding işlemleri optimize edilmiş bir native motor üzerinden yürütülebilir.

Rayon, CPU paralelliğinin yönetilmesinde kullanılır.

### 5.2 CPU ve GPU ayrımı

Mevcut V13.1 eğitim kodunun çalışma biçimi:

- ByteLevel BPE eğitimi CPU üzerinde yürütülür.
- Tokenizer motoru Rust/Rayon tabanlıdır.
- Zeyrek ve Morfessor kendi analiz süreçlerini yürütür.
- GPU kullanımı mevcut kodda etkin değildir.

Bu nedenle GPU donanımının sistemde bulunması ile tokenizer eğitiminin GPU üzerinde gerçekleştirilmesi birbirinden ayrı kavramlardır.

---

## 6. Eğitim Verisi ve Donanım

Proje için bildirilen toplam eğitim verisi:

<details>
<summary>Training configuration</summary>

| Özellik | Değer |
|---|---:|
| Toplam veri | **304,92 GB** |
| Yaklaşık süre | **1,5 gün** |
| Hedef vocabulary | **256.000** |
| RAM kapasitesi | **1 TB** |
| Depolama kapasitesi | **10 TB** |
| GPU donanımı | NVIDIA RTX 6000 PRO Blackwell |
| BPE algoritması | ByteLevel BPE |
| Minimum token frequency | 2 |

</details>

Bu değerler proje geliştirme süreci için bildirilen yapılandırmadır. Gerçek eğitim süresi ve donanım kullanımı çalıştırma logları ve sonuç dosyalarıyla doğrulanmalıdır.

Özellikle mevcut V13.1 kodunda BPE eğitiminin CPU üzerinde çalıştığı ve `gpu_used` değerinin `false` olarak raporlandığı unutulmamalıdır.

---

## 7. 12 Aşamalı Geliştirme Pipeline'ı

Proje tek bir eğitim komutundan ibaret değildir. Modüllere ayrılmış bir geliştirme sistemine sahiptir.

Genel akış:

1. Paket kurulumu ve ortam kontrolü.
2. Dataset kaynaklarının keşfi.
3. Dosyaların indirilmesi.
4. Parquet dosyalarının doğrulanması.
5. Veri akışı ve belge hazırlama.
6. Veri örnekleme ve istatistik toplama.
7. Kelime havuzunun oluşturulması.
8. Zeyrek morfolojik çözümleme.
9. Morfessor eğitimi ve segmentasyon.
10. Birleşik morfoloji analizi.
11. ByteLevel BPE eğitimi ve benchmark.
12. Sonuçların doğrulanması, README ve ZIP paketinin oluşturulması.

Her modül kendi durumunu ve çalışma süresini raporlayabilir.

### 7.1 Resumable pipeline

Pipeline içerisinde daha önce tamamlanan bazı aşamaların yeniden kullanılmasını sağlayan imza ve cache sistemleri bulunur.

Bu sayede aynı girdilerle tekrar çalıştırıldığında geçerli sonuçların yeniden hesaplanması önlenebilir.

### 7.2 Atomic file operations

Önemli çıktı dosyaları önce geçici dosyalara yazılır ve işlem başarılı olduğunda hedef dosyaya taşınır.

Bu yaklaşım yarım kalmış çıktıların geçerli dosyaların üzerine yazılmasını önlemeye yardımcı olur.

### 7.3 Logging ve monitoring

Sistem:

- CPU kullanımını,
- RAM kullanımını,
- Disk durumunu,
- İşlem sürelerini,
- Veri miktarını,
- Modül durumlarını,
- Hata kayıtlarını

izleyebilir.

Uzun süren BPE ve Morfessor işlemlerinde heartbeat sistemiyle çalışma durumunun düzenli olarak raporlanması da mümkündür.

---

## 8. Benchmark Sistemi

RuzgarTokenizer, token sayısı, hız, karakter verimliliği ve round-trip doğruluğu gibi farklı metriklerle değerlendirilebilir.

### Benchmark metrikleri

- Total token count.
- Tokens per word.
- Characters per token.
- Bytes per token.
- Tokens per 1,000 characters.
- Documents per second.
- Tokens per second.
- Batch throughput.
- Single-thread throughput.
- Round-trip accuracy.
- Word fertility.
- Morphological boundary alignment.

### 8.1 Karşılaştırma yaklaşımı

Geliştirme sürecinde NEDO Tokenizer ve Morpheus tasarım ve karşılaştırma referansları olarak ele alınmıştır.

Ek benchmark karşılaştırmalarında OpenAI'nin public encoding'leri, Qwen ve DeepSeek tokenizer'ları da değerlendirilmiştir.

Açıkça belirtilmesi gereken nokta: OpenAI encoding'leri bütün ChatGPT modellerinin birebir tokenizer'ı değildir. Gemini ve Claude'un üretim tokenizer'ları da bu karşılaştırmada doğrudan ölçülmüş kabul edilmemelidir.

---

## 9. Benchmark Sonuçları

Aşağıdaki tablo, proje için sağlanan 200 MiB benchmark raporundaki sonuçları göstermektedir.

| Sıralama | Tokenizer | Toplam token |
|---:|---|---:|
| 1 | **Ruzgar 256K** | **35.741.100** |
| 2 | OpenAI o200k_base | 57.460.095 |
| 3 | Qwen 2.5 | 66.276.713 |
| 4 | OpenAI cl100k_base | 75.408.571 |
| 5 | DeepSeek V3 | 79.031.654 |
| 6 | NEDO 32K | 81.762.570 |

### 9.1 NEDO karşılaştırması

| Ölçüm | Sonuç |
|---|---:|
| NEDO token sayısı | 81.762.570 |
| Ruzgar token sayısı | 35.741.100 |
| Fark | 46.021.470 |
| Token sayısındaki azalma | **%56,29** |

Aynı test metinlerinde Ruzgar, NEDO'ya kıyasla %56,29 daha az token üretmiştir.

### 9.2 Diğer sistemlerle karşılaştırma

| Tokenizer | Ruzgar'ın daha az token üretme oranı |
|---|---:|
| NEDO 32K | %56,29 |
| OpenAI o200k_base | %37,80 |
| Qwen 2.5 | %46,08 |
| OpenAI cl100k_base | %52,61 |
| DeepSeek V3 | %54,77 |

### 9.3 Token / Byte

| Tokenizer | Tokens / Byte |
|---|---:|
| **Ruzgar 256K** | **0,1704** |
| OpenAI o200k_base | 0,2740 |
| Qwen 2.5 | 0,3160 |
| OpenAI cl100k_base | 0,3596 |
| DeepSeek V3 | 0,3769 |
| NEDO 32K | 0,3899 |

### 9.4 Token / 1000 kelime

| Tokenizer | Tokens / 1000 words |
|---|---:|
| **Ruzgar 256K** | **1.452,18** |
| OpenAI o200k_base | 2.334,64 |
| Qwen 2.5 | 2.692,86 |
| OpenAI cl100k_base | 3.063,89 |
| DeepSeek V3 | 3.211,10 |
| NEDO 32K | 3.322,06 |

### 9.5 Tokenization throughput

| Tokenizer | Tokens / second |
|---|---:|
| OpenAI cl100k_base | 4.291.790 |
| OpenAI o200k_base | 4.079.031 |
| DeepSeek V3 | 1.092.013 |
| Qwen 2.5 | 897.627 |
| NEDO 32K | 554.285 |
| Ruzgar 256K | 466.251 |

Ruzgar'ın ana avantajı bu testte en yüksek hız değil, daha düşük token üretimidir.

### 9.6 Latency

| Tokenizer | P50 | P95 | P99 |
|---|---:|---:|---:|
| Ruzgar | 0,662 ms | 4,634 ms | 11,256 ms |
| NEDO | 1,266 ms | 9,160 ms | 22,525 ms |
| OpenAI o200k | 0,120 ms | 0,768 ms | 1,780 ms |
| OpenAI cl100k | 0,151 ms | 0,994 ms | 2,303 ms |
| Qwen 2.5 | 0,653 ms | 4,447 ms | 10,422 ms |
| DeepSeek V3 | 0,626 ms | 4,430 ms | 10,363 ms |

### 9.7 Benchmark değerlendirmesi

Bu sonuçlar, sağlanan benchmark raporundaki test metinleri ve tokenizer sürümleri için geçerlidir.

Token sayısının düşük olması tek başına her görevde daha iyi performans anlamına gelmez. Round-trip doğruluğu, tokenization hızı, farklı alanlardaki verimlilik ve morfolojik uyum da değerlendirilmelidir.

---

## 10. Doğrulama ve Kalite Kontrolü

Tokenizer eğitiminin ardından çeşitli testler uygulanabilir.

### Round-trip validation

Metnin encode ve decode işlemlerinden sonra orijinal metinle aynı olup olmadığı kontrol edilir.

### Synthetic tests

Türkçe karakterler, ekler, kesme işaretleri, özel karakterler ve farklı metin biçimleriyle sentetik testler yapılır.

### Holdout tests

Eğitim verisinden ayrılan belgeler üzerinde değerlendirme yapılabilir.

### Batch consistency

Batch encoding ve tek tek encoding sonuçlarının tutarlılığı kontrol edilir.

### Save/load validation

Kaydedilen tokenizer yeniden yüklenerek aynı metinlerde aynı token ID dizilerini üretip üretmediği test edilir.

---

## 11. Dosya Yapısı ve Çıktılar

Projenin ana çıktıları aşağıdaki gibi düzenlenmiştir:

```text
RUZGAR_V13/
│
├── datasets/
│   ├── fineweb2/
│   ├── turkish_corpus_100b/
│   ├── umarigan/
│   └── _cache/
│
├── models/
│   └── ruzgar_256k/
│       ├── tokenizer.json
│       └── config.json
│
├── morphology/
│   ├── morphology_word_corpus.tsv
│   ├── zeyrek_analyses.jsonl
│   ├── morfessor_model.bin
│   ├── morfessor_segments.jsonl
│   ├── unified_morphology.jsonl
│   └── suffix_inventory.json
│
├── benchmarks/
│   └── benchmark_results.json
│
├── logs/
│   └── ruzgar_v13_1_*.log
│
├── export/
│   └── RuzgarTokenizer.zip
│
├── results.json
└── README.md
```

Dosyaların gerçek isimleri ve mevcut olup olmadıkları çalıştırma yapılandırmasına göre değişebilir.

---

## 12. GitHub ve Hugging Face Dağıtımı

Tokenizer'ın büyük dosyaları standart GitHub dosya yükleme sınırlarını aşabildiğinden proje dosyaları **ZIP arşivi** halinde GitHub üzerinde dağıtılmaktadır.

Bu yöntem, büyük vocabulary ve model dosyalarının tek bir paket olarak indirilmesini kolaylaştırır.

### Hugging Face

Tokenizer dosyaları:

https://huggingface.co/brun12/RuzgarTokenizer

### GitHub

Kaynak kodu, README, benchmark raporları ve ZIP dağıtım paketi:

`RuzgarTokenizer`

ZIP arşivi indirilirken sürüm ve dosya bütünlüğü bilgilerinin kontrol edilmesi önerilir.

---

## 13. Kullanım

Python ile yerel tokenizer dosyasını yüklemek için:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)

text = "Türkçenin morfolojik yapısını verimli biçimde temsil etmek."

encoded = tokenizer.encode(text)

print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
print("Token count:", len(encoded.ids))

decoded = tokenizer.decode(encoded.ids)

print("Decoded:", decoded)
```

### Tokenizer bilgilerini görüntüleme

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Special tokens:", tokenizer.get_added_tokens_decoder())
```

---

## 14. Gelecek Geliştirmeler

- Daha geniş Türkçe benchmark kümeleri.
- Farklı alanlarda token verimliliği karşılaştırmaları.
- Daha kapsamlı morfolojik değerlendirme.
- BPE ve morfem sınırlarının daha ayrıntılı incelenmesi.
- Encoding ve decoding hız optimizasyonları.
- Bellek ve model yükleme süresi ölçümleri.
- Uzun metin performans testleri.
- Daha fazla tokenizer karşılaştırması.
- Python paket entegrasyonu.
- Daha kapsamlı API ve kullanım dokümantasyonu.

---

## 15. Geliştirici

**Rüzgar Vatansever**

RuzgarTokenizer, Türkçe için daha verimli tokenizasyon sistemleri geliştirmek amacıyla hazırlanmış bağımsız bir projedir.

Projenin odağında Türkçenin morfolojik ve dilbilimsel özellikleri, büyük ölçekli veri işleme, vocabulary tasarımı ve karşılaştırmalı tokenizer araştırmaları bulunmaktadır.

**Project:** RuzgarTokenizer  
**Developer:** Rüzgar Vatansever  
**Algorithm:** ByteLevel BPE  
**Vocabulary target:** 256K  
**Primary language:** Turkish  
**Hugging Face:** https://huggingface.co/brun12/RuzgarTokenizer

---

# 🇬🇧 ENGLISH

## 1. About the Project

**RuzgarTokenizer** is an experimental Turkish-focused tokenizer project developed by **Rüzgar Vatansever**. It features a target vocabulary of 256,000 tokens and is designed to investigate efficient text representation for Turkish, with particular attention to the language's agglutinative morphology and word-formation patterns.

The project goes beyond basic tokenizer training. It includes a complete development pipeline covering large-scale dataset processing, corpus validation, Turkish-specific pre-tokenization, morphological analysis, unsupervised morpheme discovery, vocabulary construction, BPE training, tokenizer validation, and comparative benchmarking.

### Key features

- 256,000 target vocabulary.
- ByteLevel BPE algorithm.
- Rust-powered Hugging Face Tokenizers engine.
- Custom Turkish pre-tokenization.
- Zeyrek morphological analysis.
- Morfessor unsupervised morpheme discovery.
- Custom Turkish suffix inventory.
- Morphological analysis and BPE boundary alignment.
- Large-scale Parquet processing.
- Parallel CPU and Rust/Rayon execution.
- Round-trip and consistency validation.
- Comparative tokenizer benchmarking.

---

## 2. Tokenizer Architecture

### 2.1 ByteLevel BPE

RuzgarTokenizer is built around **ByteLevel Byte Pair Encoding (BPE)**.

BPE learns frequently occurring character sequences and subword units from a training corpus. These learned units are represented in the tokenizer vocabulary.

The ByteLevel approach operates on byte-level text representations, helping the tokenizer handle different characters and writing patterns.

Reasons for this design include:

- Byte-level representation of Turkish characters.
- Flexible subword tokenization.
- Handling of previously unseen character sequences.
- Efficient representation of frequent text patterns with a large vocabulary.
- Compatibility with modern NLP infrastructure.

The actual algorithm is **ByteLevel BPE, not Unigram**.

### 2.2 Custom Turkish Pre-tokenizer

Generic pre-tokenization rules do not always account for the writing conventions and morphological structure of every language.

Turkish introduces additional considerations, including suffixation, apostrophes, proper nouns, and language-specific characters.

RuzgarTokenizer therefore includes a custom pre-tokenization mode called `turkish_split`.

It aims to provide more controlled initial text segmentation based on Turkish writing conventions and word boundaries.

An alternative GPT-2 regex-based pre-tokenization mode is also available.

### 2.3 Vocabulary configuration

The target vocabulary size is:

**256,000 tokens**

| Parameter | Value |
|---|---:|
| Target vocabulary | 256,000 |
| Minimum token frequency | 2 |
| Special tokens | 2 |
| Special tokens | `<|endoftext|>`, `<|pad|>` |
| Default pre-tokenizer | Turkish Split |
| Alternative pre-tokenizer | GPT-2 Regex |
| Maximum token length | Configurable |

The actual vocabulary size is checked after training. The requested vocabulary and the final vocabulary are treated as separate values.

---

## 3. Turkish Morphology Research

Turkish is an agglutinative language. A single word stem can receive multiple derivational and inflectional suffixes.

For example:

`ev → evler → evlerimiz → evlerimizden → evlerimizdekiler`

This characteristic makes Turkish tokenizer development different from approaches designed primarily around English text.

RuzgarTokenizer includes a dedicated morphology research pipeline.

### 3.1 Zeyrek — Morphological Analysis

**Zeyrek** is used to analyze Turkish words and their possible morphological structures.

Its role in the project includes:

- Investigating word roots.
- Extracting possible morphological analyses.
- Studying suffix structures.
- Examining ambiguous words with multiple analyses.
- Analyzing part-of-speech distributions.
- Evaluating lemma frequencies.

Raw Zeyrek analyses and summary statistics can be stored as structured output.

Zeyrek is a morphological analyzer, not a tokenizer. It is part of the project's linguistic research pipeline.

### 3.2 Morfessor — Unsupervised Morpheme Discovery

**Morfessor** is used for unsupervised morpheme discovery and segmentation.

It helps investigate how words can be divided into smaller recurring linguistic units.

Its project applications include:

- Learning possible morpheme boundaries.
- Discovering frequent morphemes.
- Producing frequency-weighted morpheme statistics.
- Segmenting Turkish word forms.
- Providing additional linguistic evidence for tokenizer research.

The trained Morfessor model is saved and reloaded for validation.

### 3.3 Custom Turkish suffix inventory

The project includes a custom suffix inventory for examining Turkish suffix structures.

The inventory covers categories such as:

- Inflectional suffixes.
- Denominal derivational suffixes.
- Deverbal derivational suffixes.
- Denominal verb-forming suffixes.
- Verbal derivational suffixes.
- Participial structures.
- Converb structures.
- Nominalization patterns.

Surface forms can be generated while accounting for Turkish vowel harmony, consonant alternation, and buffer consonants.

Examples include:

- `-lar / -ler`
- `-cı / -ci / -cu / -cü`
- `-lık / -lik / -luk / -lük`
- `-dan / -den / -tan / -ten`

This system provides additional linguistic analysis capabilities.

### 3.4 Unified morphology analysis

Zeyrek and Morfessor outputs can be combined into a unified morphology dataset.

The combined records can include:

- Zeyrek analyses.
- Morfessor segmentations.
- Word frequencies.
- Custom suffix inventory results.
- Heuristic suffix analysis.
- Cross-check information.

This makes it possible to examine how different morphological approaches interpret the same word.

### 3.5 BPE-to-morphology alignment

One of the project's research components compares BPE token boundaries with morpheme boundaries produced by Morfessor.

The evaluation can calculate:

- BPE boundary precision.
- Morfessor boundary recall.
- F1 score.

These measurements help investigate how closely the learned BPE segmentation corresponds to linguistic morpheme boundaries.

This is a research and evaluation component. Zeyrek and Morfessor are not required to run during every tokenizer inference operation.

---

## 4. Dataset Infrastructure

RuzgarTokenizer is designed to process large Turkish text corpora.

### Dataset sources

| Dataset | Purpose |
|---|---|
| FineWeb-2 Turkish | Broad Turkish web text |
| Turkish Corpus 100B | Large-scale Turkish text distribution |
| Umarigan Turkish Corpus | Additional Turkish corpus diversity |

Configured source weights:

| Source | Weight |
|---|---:|
| FineWeb-2 Turkish | 40% |
| Turkish Corpus 100B | 40% |
| Umarigan Turkish Corpus | 20% |

These are configured source weights, not a guarantee that the final training corpus will contain exactly the same proportions.

### 4.1 Hugging Face Hub

Hugging Face Hub is used for dataset discovery and file downloads.

The pipeline selects dataset files according to configured repositories and file patterns.

### 4.2 PyArrow and Parquet

PyArrow is a core part of the data-processing infrastructure.

It enables:

- Reading Parquet metadata.
- Inspecting dataset schemas.
- Detecting text columns.
- Reading row groups and batches.
- Processing large datasets without loading every file into memory at once.

This approach makes processing hundreds of gigabytes of corpus data more manageable.

### 4.3 Dataset validation

The pipeline includes:

- File integrity checks.
- Parquet schema validation.
- Text-column detection.
- Language-label checks.
- Invalid-file handling.
- Download retries.
- Download and validation reports.

### 4.4 Streaming and batch processing

Documents are processed through a streaming reader and grouped into configurable batches.

Instead of loading an entire corpus into RAM, the pipeline processes data incrementally.

### 4.5 Deduplication

An optional Bloom Filter-based duplicate detection mechanism is included.

It can identify exact duplicate text entries.

A Bloom Filter is probabilistic: it does not independently identify near-duplicates or semantically similar documents.

### 4.6 Holdout sampling

The pipeline supports keeping a separate holdout sample outside the BPE training stream.

This makes it possible to evaluate tokenizer behavior on documents that were not used during training.

---

## 5. Training Architecture and Technologies

RuzgarTokenizer combines multiple technologies into a modular NLP development system.

| Technology | Project role |
|---|---|
| Hugging Face Hub | Dataset and model artifact management |
| Hugging Face Tokenizers | Tokenizer implementation and training |
| Rust / Rayon | Parallel CPU execution in the tokenizer engine |
| ByteLevel BPE | Main tokenization algorithm |
| PyArrow | Parquet reading and processing |
| Zeyrek | Turkish morphological analysis |
| Morfessor | Unsupervised morpheme learning |
| Python | Pipeline orchestration and analysis |
| SQLite | Morphology indexing and queries |
| JSON / JSONL | Structured result storage |
| Multiprocessing | Parallel analysis workers |
| ThreadPoolExecutor | Parallel tasks and file operations |
| psutil | CPU, RAM, and disk monitoring |
| hashlib | File and data signatures |
| Bloom Filter | Optional duplicate detection |
| Marimo | Notebook-based execution environment |

### 5.1 Rust-powered Tokenizers engine

Hugging Face Tokenizers is implemented in Rust.

This provides a native, optimized engine for tokenizer training and encoding operations.

Rayon is used for CPU parallelism in the tokenizer engine.

### 5.2 CPU and GPU distinction

The current V13.1 implementation has the following execution characteristics:

- ByteLevel BPE training runs on the CPU.
- The tokenizer engine uses Rust/Rayon-based processing.
- Zeyrek and Morfessor perform their respective analysis workloads.
- GPU execution is not enabled in the current implementation.

The presence of GPU hardware in the development environment should therefore be distinguished from actual GPU utilization during tokenizer training.

---

## 6. Training Data and Hardware

The reported training configuration:

| Specification | Value |
|---|---:|
| Total training data | **304.92 GB** |
| Approximate duration | **1.5 days** |
| Target vocabulary | **256,000** |
| RAM capacity | **1 TB** |
| Storage capacity | **10 TB** |
| GPU hardware | NVIDIA RTX 6000 PRO Blackwell |
| BPE algorithm | ByteLevel BPE |
| Minimum token frequency | 2 |

These figures describe the project-reported development environment and training run. Actual runtime and hardware utilization should be verified against execution logs and generated reports.

In particular, the current V13.1 implementation reports BPE training as CPU-based and records `gpu_used` as `false`.

---

## 7. The 12-Stage Development Pipeline

RuzgarTokenizer is organized as a modular development system rather than a single training command.

The overall workflow consists of:

1. Package installation and environment checks.
2. Dataset source discovery.
3. Dataset file downloads.
4. Parquet validation.
5. Document streaming and preparation.
6. Data sampling and statistics collection.
7. Word corpus construction.
8. Zeyrek morphological analysis.
9. Morfessor training and segmentation.
10. Unified morphology analysis.
11. ByteLevel BPE training and benchmarking.
12. Result validation, README generation, and ZIP packaging.

Each module can report its own execution status and runtime.

### 7.1 Resumable pipeline

The pipeline includes signatures and caching mechanisms that allow valid outputs from completed stages to be reused.

When the inputs have not changed, unnecessary recomputation can be avoided.

### 7.2 Atomic file operations

Important output files are first written to temporary files and moved into their final locations only after successful completion.

This helps protect valid outputs from being overwritten by incomplete files.

### 7.3 Logging and monitoring

The system can monitor:

- CPU utilization.
- RAM usage.
- Disk status.
- Execution time.
- Processed data volume.
- Module status.
- Error records.

Heartbeat reporting is also available for long-running operations such as BPE and Morfessor training.

---

## 8. Benchmark System

RuzgarTokenizer can be evaluated using multiple metrics rather than vocabulary size alone.

### Supported benchmark metrics

- Total token count.
- Tokens per word.
- Characters per token.
- Bytes per token.
- Tokens per 1,000 characters.
- Documents per second.
- Tokens per second.
- Batch throughput.
- Single-thread throughput.
- Round-trip accuracy.
- Word fertility.
- Morphological boundary alignment.

### 8.1 Comparison approach

NEDO Tokenizer and Morpheus were considered important design and comparison references during development.

Additional benchmark comparisons include publicly available OpenAI encodings, Qwen, and DeepSeek tokenizers.

OpenAI public encodings should not be interpreted as the exact tokenizer of every ChatGPT model. Gemini and Claude production tokenizers should not be considered directly tested in this benchmark.

---

## 9. Benchmark Results

The following results are from the supplied 200 MiB benchmark report.

| Rank | Tokenizer | Total tokens |
|---:|---|---:|
| 1 | **Ruzgar 256K** | **35,741,100** |
| 2 | OpenAI o200k_base | 57,460,095 |
| 3 | Qwen 2.5 | 66,276,713 |
| 4 | OpenAI cl100k_base | 75,408,571 |
| 5 | DeepSeek V3 | 79,031,654 |
| 6 | NEDO 32K | 81,762,570 |

### 9.1 Comparison with NEDO

| Measurement | Result |
|---|---:|
| NEDO token count | 81,762,570 |
| Ruzgar token count | 35,741,100 |
| Difference | 46,021,470 |
| Reduction in token count | **56.29%** |

On the same benchmark text, Ruzgar generated 56.29% fewer tokens than NEDO.

### 9.2 Comparison with other systems

| Tokenizer | Fewer tokens generated by Ruzgar |
|---|---:|
| NEDO 32K | 56.29% |
| OpenAI o200k_base | 37.80% |
| Qwen 2.5 | 46.08% |
| OpenAI cl100k_base | 52.61% |
| DeepSeek V3 | 54.77% |

### 9.3 Tokens per byte

| Tokenizer | Tokens / Byte |
|---|---:|
| **Ruzgar 256K** | **0.1704** |
| OpenAI o200k_base | 0.2740 |
| Qwen 2.5 | 0.3160 |
| OpenAI cl100k_base | 0.3596 |
| DeepSeek V3 | 0.3769 |
| NEDO 32K | 0.3899 |

### 9.4 Tokens per 1,000 words

| Tokenizer | Tokens / 1,000 words |
|---|---:|
| **Ruzgar 256K** | **1,452.18** |
| OpenAI o200k_base | 2,334.64 |
| Qwen 2.5 | 2,692.86 |
| OpenAI cl100k_base | 3,063.89 |
| DeepSeek V3 | 3,211.10 |
| NEDO 32K | 3,322.06 |

### 9.5 Tokenization throughput

| Tokenizer | Tokens / second |
|---|---:|
| OpenAI cl100k_base | 4,291,790 |
| OpenAI o200k_base | 4,079,031 |
| DeepSeek V3 | 1,092,013 |
| Qwen 2.5 | 897,627 |
| NEDO 32K | 554,285 |
| Ruzgar 256K | 466,251 |

Ruzgar's primary advantage in this benchmark is lower token usage, not the highest raw processing speed.

### 9.6 Latency

| Tokenizer | P50 | P95 | P99 |
|---|---:|---:|---:|
| Ruzgar | 0.662 ms | 4.634 ms | 11.256 ms |
| NEDO | 1.266 ms | 9.160 ms | 22.525 ms |
| OpenAI o200k | 0.120 ms | 0.768 ms | 1.780 ms |
| OpenAI cl100k | 0.151 ms | 0.994 ms | 2.303 ms |
| Qwen 2.5 | 0.653 ms | 4.447 ms | 10.422 ms |
| DeepSeek V3 | 0.626 ms | 4.430 ms | 10.363 ms |

### 9.7 Benchmark interpretation

These results apply to the benchmark text and tokenizer versions used in the supplied report.

A lower token count alone does not establish better performance for every task. Round-trip accuracy, encoding speed, domain-specific efficiency, and morphological alignment should also be evaluated.

---

## 10. Validation and Quality Assurance

Several validation procedures can be performed after tokenizer training.

### Round-trip validation

Checks whether encoding and decoding preserve the original text.

### Synthetic tests

Tests Turkish characters, suffixes, apostrophes, special characters, and different text formats.

### Holdout tests

Evaluates the tokenizer on documents separated from the training stream.

### Batch consistency

Checks whether batch encoding produces results consistent with individual encoding.

### Save/load validation

Reloads the saved tokenizer and verifies that it produces consistent token IDs for the same inputs.

---

## 11. Project Structure and Outputs

The main project artifacts are organized approximately as follows:

```text
RUZGAR_V13/
│
├── datasets/
│   ├── fineweb2/
│   ├── turkish_corpus_100b/
│   ├── umarigan/
│   └── _cache/
│
├── models/
│   └── ruzgar_256k/
│       ├── tokenizer.json
│       └── config.json
│
├── morphology/
│   ├── morphology_word_corpus.tsv
│   ├── zeyrek_analyses.jsonl
│   ├── morfessor_model.bin
│   ├── morfessor_segments.jsonl
│   ├── unified_morphology.jsonl
│   └── suffix_inventory.json
│
├── benchmarks/
│   └── benchmark_results.json
│
├── logs/
│   └── ruzgar_v13_1_*.log
│
├── export/
│   └── RuzgarTokenizer.zip
│
├── results.json
└── README.md
```

The exact files present depend on the completed stages and configuration of each run.

---

## 12. GitHub and Hugging Face Distribution

Because the tokenizer vocabulary and associated model assets can exceed standard GitHub file-upload limits, the large project files are distributed as a **ZIP archive on GitHub**.

This makes the artifacts easier to download as a single package.

### Hugging Face

Tokenizer files:

https://huggingface.co/brun12/RuzgarTokenizer

### GitHub

Source code, README, benchmark reports, and ZIP distribution package:

`RuzgarTokenizer`

Users should check release information and file integrity when downloading the archive.

---

## 13. Usage

Load the local tokenizer with Python:

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(
    "tokenizer.json"
)

text = "Efficiently representing the morphological structure of Turkish."

encoded = tokenizer.encode(text)

print("Tokens:", encoded.tokens)
print("Token IDs:", encoded.ids)
print("Token count:", len(encoded.ids))

decoded = tokenizer.decode(encoded.ids)

print("Decoded:", decoded)
```

### Inspect tokenizer information

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("tokenizer.json")

print("Vocabulary size:", tokenizer.get_vocab_size())
print("Special tokens:", tokenizer.get_added_tokens_decoder())
```

---

## 14. Future Development

Potential future improvements include:

- Larger Turkish benchmark datasets.
- Domain-specific token-efficiency evaluations.
- More extensive morphological evaluation.
- Detailed BPE-to-morpheme boundary analysis.
- Encoding and decoding performance optimization.
- Memory and loading-time measurements.
- Long-text performance testing.
- Additional tokenizer comparisons.
- Python package integration.
- Expanded API and usage documentation.

---

## 15. Developer

**Rüzgar Vatansever**

RuzgarTokenizer is an independent project created to investigate more efficient tokenization for Turkish.

Its focus is on Turkish linguistic structure, large-scale data processing, vocabulary design, and comparative tokenizer research.

**Project:** RuzgarTokenizer  
**Developer:** Rüzgar Vatansever  
**Algorithm:** ByteLevel BPE  
**Target vocabulary:** 256K  
**Primary language:** Turkish  
**Hugging Face:** https://huggingface.co/brun12/RuzgarTokenizer

---

## Attribution

If you use RuzgarTokenizer in research or software, attribution to **Rüzgar Vatansever** and a reference to the official repository are appreciated.

Please consult the repository's license before redistribution or commercial integration.
