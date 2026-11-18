# RuzgarTokenizer
### A 256K Turkish-Focused Tokenization & Morphological Research Framework

**Developed by Rüzgar Vatansever**

[Hugging Face Model Repository](https://huggingface.co/brun12/RuzgarTokenizer)

Download: https://huggingface.co/brun12/RuzgarTokenizer/tree/main

---

# TÜRKÇE DOKÜMANTASYON

## 1. Proje Hakkında

**RuzgarTokenizer**, Türkçe metinlerde token verimliliğini, alt birim dağılımını ve morfolojik yapıyı incelemek amacıyla geliştirilen, 256.000 tokenlık hedef söz dağarcığına sahip bir ByteLevel BPE tokenizer projesidir.

Projenin temel hedefi yalnızca bir tokenizer eğitmek değildir. Aynı zamanda büyük ölçekli metin koleksiyonlarını işleyebilen, Türkçeye özel ön işleme yöntemlerini değerlendirebilen, morfolojik analizleri farklı yaklaşımlarla karşılaştırabilen ve sonuçlarını ölçülebilir benchmark verileriyle raporlayabilen modüler bir araştırma altyapısı oluşturmaktır.

RuzgarTokenizer'ın geliştirilmesinde NEDO ve Morpheus gibi tokenizer projeleri karşılaştırmalı referanslar olarak değerlendirilmiştir. Bunun yanında OpenAI'nin kamuya açık tokenization encodings, Qwen ve DeepSeek gibi sistemlerin tokenizasyon sonuçları da performans karşılaştırmalarında incelenmiştir.

Proje, Türkçenin eklemeli dil yapısını, kelime türetme ve çekim süreçlerini, uzun kelime biçimlerini ve karakter dağılımlarını dikkate alan bir geliştirme yaklaşımı benimser.

### Temel özellikler

- 256.000 hedef vocabulary kapasitesi.
- Rust tabanlı Hugging Face Tokenizers altyapısı.
- ByteLevel BPE tokenization.
- Türkçe odaklı özel pre-tokenization yaklaşımı.
- Zeyrek ile Türkçe biçimbilimsel analiz.
- Morfessor ile denetimsiz morfem keşfi.
- Özel Türkçe ek envanteri ve sezgisel morfolojik analiz.
- Büyük ölçekli Parquet veri kümeleriyle çalışma.
- Streaming ve batch tabanlı veri işleme.
- Çok iş parçacıklı indirme ve paralel işlem altyapısı.
- SQLite tabanlı ara indeksleme.
- JSON ve JSONL tabanlı araştırma çıktıları.
- Token fertility, round-trip ve throughput ölçümleri.
- BPE–Morfessor sınır hizalama analizi.
- Otomatik raporlama, loglama ve ZIP paketleme.

---

## 2. Tokenizer Architecture

### 2.1 ByteLevel BPE

RuzgarTokenizer'ın temel tokenization algoritması **Byte-Level Byte Pair Encoding (BPE)** yöntemidir.

BPE, metin içindeki sık karşılaşılan sembol dizilerini birleştirerek daha büyük token birimleri oluşturur. Eğitim sürecinde bu birleşimler, metin korpusundaki frekans dağılımlarından öğrenilir.

Byte-level yaklaşım, metinleri byte temelli bir gösterim üzerinden ele alarak farklı karakterlerin ve Unicode metinlerin işlenmesinde esneklik sağlar.

Bu yaklaşım özellikle Türkçe gibi ASCII dışı karakterlerin yoğun kullanıldığı diller açısından önemlidir.

Örnek Türkçe karakterler:

- ç
- ğ
- ı
- İ
- ö
- ş
- ü

RuzgarTokenizer, bu karakterleri içeren metinlerin tokenization sürecini ByteLevel altyapısı üzerinden gerçekleştirir.

### 2.2 Rust-Powered Tokenization Engine

Tokenizer eğitim ve encoding altyapısında Hugging Face `tokenizers` kütüphanesi kullanılmaktadır.

Bu kütüphane, performans açısından kritik işlemleri Rust tarafında gerçekleştirir. Python ise veri hazırlama, eğitim akışını yönetme, morfolojik analizleri yürütme, raporlama ve deney organizasyonu gibi üst seviye görevleri üstlenir.

Bu mimari, Python'ın esnekliğini Rust'ın performans odaklı işlem altyapısıyla birleştirir.

Kullanılan teknolojiler:

- **Rust:** Tokenizer motorunun performans odaklı uygulaması.
- **Rayon:** Rust tarafındaki paralel işlem altyapısı.
- **Python:** Veri işleme, deney yönetimi ve orkestrasyon.
- **Hugging Face Tokenizers:** BPE modelinin eğitimi ve tokenizer operasyonları.

Kod, paralellik için `TOKENIZERS_PARALLELISM`, `RAYON_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` ve `OPENBLAS_NUM_THREADS` gibi ortam değişkenlerini yapılandırır.

**Önemli:** Bu mimari GPU eğitimi anlamına gelmez. V13.2 kodu BPE eğitimini CPU üzerinde Rust/Rayon altyapısıyla yürütür. GPU kullanımı bu sürümde etkin değildir.

### 2.3 Vocabulary Configuration

| Parametre | Değer |
|---|---:|
| Hedef vocabulary | 256.000 |
| BPE minimum frequency | 2 |
| Special token | `<|endoftext|>` |
| Padding token | `<|pad|>` |
| Tokenization algorithm | ByteLevel BPE |
| Default pre-tokenizer mode | `turkish_split` |

256K vocabulary tercihi, Türkçe kelime biçimlerinin ve sık karşılaşılan alt dizilerin daha geniş bir token havuzunda temsil edilmesini araştırmak amacıyla yapılmıştır.

Büyük vocabulary tek başına daha iyi bir tokenizer garantisi değildir. Gerçek performans; token sayısı, modelin embedding maliyeti, genelleme, latency, throughput ve farklı metin türlerindeki davranış birlikte değerlendirilerek yorumlanmalıdır.

---

## 3. Turkish-Aware Pre-Tokenization

Türkçe, kelime köklerine çok sayıda yapım ve çekim eki eklenebilen eklemeli bir dildir.

Örneğin tek bir kök, farklı ek kombinasyonlarıyla çok sayıda kelime biçimi oluşturabilir.

Bu durum tokenizer tasarımında şu problemleri ortaya çıkarabilir:

- Uzun kelimelerin fazla sayıda parçaya ayrılması.
- Sık kullanılan eklerin farklı bağlamlarda tutarsız temsil edilmesi.
- Kelime kökleri ile eklerin token sınırlarının örtüşmemesi.
- Türkçe karakterlerin ve apostrof kullanımının işlenmesi.
- Nadir kelime biçimlerinin token dağılımına etkisi.

RuzgarTokenizer bu konuları incelemek için iki pre-tokenization seçeneği sunar:

1. `turkish_split`
2. `gpt2`

Varsayılan yapılandırma:

```python
PRETOKENIZER_MODE = "turkish_split"
```

Özel Türkçe split yaklaşımı, BPE eğitiminden önce metnin parçalanma biçimini Türkçe metin özelliklerine göre düzenlemek amacıyla kullanılır.

GPT-2 seçeneği ise karşılaştırma ve alternatif deneyler için yapılandırmada tutulur.

Bu iki seçenek aynı tokenizer algoritmasının yerine geçen farklı algoritmalar değildir. BPE eğitimine giden metnin önceden nasıl bölüneceğini belirleyen seçeneklerdir.

---

## 4. Morphological Research Architecture

RuzgarTokenizer'ın önemli araştırma bileşenlerinden biri morfolojik analiz altyapısıdır.

Bu katman, tokenizer'ın token sınırlarını Türkçenin kelime ve ek yapılarıyla karşılaştırmayı amaçlar.

Morfolojik analiz sonuçları doğrudan BPE algoritmasının yerine kullanılmaz. Bunlar ayrı araştırma çıktıları olarak üretilir ve tokenizer'ın davranışını incelemek için değerlendirilir.

### 4.1 Zeyrek — Turkish Morphological Analysis

Zeyrek, Türkçe morfolojik analiz için kullanılan bileşendir.

Proje içerisinde kelimeler üzerinde analiz gerçekleştirmek ve mümkün olduğunda aşağıdaki bilgileri elde etmek amacıyla kullanılır:

- Lemma / kök bilgisi.
- Morfolojik analiz sonuçları.
- Bir kelime için birden fazla analiz bulunup bulunmadığı.
- Analiz durumları.
- Sözcük türü (POS) dağılımları.
- Belirsiz veya çoklu analiz sonuçları.

Kod, Zeyrek işlemlerini çoklu işçi süreçleriyle yürütmeye yönelik bir altyapı da içerir.

İşlem sırasında:

- İş yükü parçalara ayrılır.
- Worker süreçleri oluşturulur.
- İşlem ilerlemesi takip edilir.
- Hata veren parçalar raporlanır.
- Gerekli durumlarda eksik parçaların ana süreçte tamamlanması sağlanır.

Zeyrek çıktıları JSONL biçiminde saklanır. Böylece her kelimenin analiz sonuçları diğer araştırma çıktılarından bağımsız olarak incelenebilir.

### 4.2 Morfessor — Unsupervised Morpheme Discovery

Morfessor, kelimeleri alt birimlere ayırmayı öğrenmek için kullanılan denetimsiz morfem keşif bileşenidir.

RuzgarTokenizer içerisinde Morfessor'ın `BaselineModel` yaklaşımı kullanılmaktadır.

Eğitim verisi, hazırlanan kelime korpusundaki kelime ve frekans bilgilerinden oluşturulur.

İşlem akışı:

1. Kelime korpusunun okunması.
2. Morfessor veri formatının hazırlanması.
3. Modelin oluşturulması.
4. `train_batch` üzerinden model eğitimi.
5. Epoch ve maliyet bilgilerinin kaydedilmesi.
6. Modelin binary biçimde diske yazılması.
7. Yeniden yüklenerek doğrulanması.
8. Viterbi tabanlı segmentasyon.
9. Segmentasyon sonuçlarının JSONL olarak saklanması.

Yapılandırılabilir parametreler:

| Parametre | Varsayılan değer |
|---|---:|
| Maximum epochs | 15 |
| Finish threshold | 0.001 |
| Damping | none |
| Top morphemes | 3.000 |

Morfessor çıktıları sayesinde şu istatistikler hesaplanabilir:

- Kelime başına ortalama morfem sayısı.
- Benzersiz morfem sayısı.
- En sık görülen morfemler.
- Frekansa göre ağırlıklandırılmış morfem dağılımı.
- Bölünmeden kalan kelimeler.
- Segmentasyon hataları.
- Segmentlerin orijinal kelimeyi yeniden oluşturup oluşturmadığı.

**Bilimsel not:** Morfessor denetimsiz bir yöntemdir. Ürettiği segmentler gerçek dilbilimsel morfemlerin kesin altın standardı olarak kabul edilmez.

### 4.3 Custom Turkish Suffix Inventory

Projede Türkçeye özgü eklerin incelenmesi için özel bir ek envanteri ve sezgisel analiz bileşenleri de bulunmaktadır.

Bu katman, kelime biçimlerini ve olası ek ayrımlarını incelemek amacıyla kullanılır.

Sezgisel analizler, Zeyrek ve Morfessor çıktılarıyla birlikte değerlendirilerek farklı yöntemlerin aynı kelime üzerindeki sonuçları karşılaştırılabilir.

### 4.4 Unified Morphology Dataset

Proje, morfolojik analiz sonuçlarını ayrı ayrı saklamakla yetinmez.

Zeyrek, Morfessor ve sezgisel analizlerden elde edilen bilgileri ortak bir veri yapısında birleştiren bir aşama da içerir.

Bu aşamada:

- JSONL dosyaları okunur.
- SQLite üzerinden kelime indeksleri oluşturulur.
- Kelime bazında analiz sonuçları eşleştirilir.
- Lemma ve segmentasyon bilgileri karşılaştırılır.
- Fonolojik değişimlere ilişkin korpus kanıtları eklenir.
- Sonuçlar birleşik JSONL veri kümesine yazılır.

SQLite ara indeksleri, tüm kayıtların aynı anda belleğe yüklenmesi ihtiyacını azaltmak amacıyla kullanılır.

### 4.5 Morphology–Tokenizer Alignment

RuzgarTokenizer, BPE token sınırlarının Morfessor tarafından üretilen segment sınırlarıyla ne ölçüde örtüştüğünü ölçen bir analiz de içerir.

Bu analizde:

- BPE token sınırları çıkarılır.
- Morfessor segment sınırları belirlenir.
- İki sınır kümesinin kesişimi hesaplanır.
- Precision, recall ve F1 değerleri üretilir.
- Tam sınır eşleşmesi görülen kelimelerin oranı hesaplanır.

Ölçümde kullanılan temel metrikler:

\[
Precision = \frac{|BPE \cap Morfessor|}{|BPE|}
\]

\[
Recall = \frac{|BPE \cap Morfessor|}{|Morfessor|}
\]

\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

Bu metrikler tokenizer'ın morfolojik açıdan kesin doğruluğunu kanıtlamaz. Morfessor segmentasyonu denetimsiz olduğu için sonuçlar, iki farklı bölümleme yaklaşımının sınır uyumunu inceleyen araştırma göstergeleri olarak yorumlanmalıdır.

---

## 5. Large-Scale Dataset Engineering

Tokenizer kalitesi, eğitim verisinin kapsamı ve niteliğinden bağımsız düşünülemez.

RuzgarTokenizer'ın veri altyapısı, Hugging Face Hub üzerinde barındırılan büyük Parquet veri kümelerini keşfetmek, indirmek, doğrulamak ve akış halinde işlemek üzere tasarlanmıştır.

### 5.1 Configured Dataset Sources

V13.2 yapılandırmasında üç veri kaynağı tanımlıdır:

| Dataset | Repository | Configured weight |
|---|---|---:|
| FineWeb-2 Turkish | `HuggingFaceFW/fineweb-2` | %40 |
| Turkish Corpus 100B | `hasankursun/turkish-corpus-100b` | %40 |
| Umarigan Turkish Corpus | `umarigan/turkish_corpus` | %20 |

Bu yüzdeler kodda tanımlanan kaynak ağırlıklarıdır. Her çalıştırmanın nihai veri dağılımının bu oranlarla birebir aynı olduğu anlamına gelmez.

Kaynaklar, dosya boyutları ve mevcut veri miktarına göre dönüşümlü biçimde işlenir.

### 5.2 Hugging Face Hub Integration

Veri keşfi ve indirme aşamasında Hugging Face Hub API kullanılır.

Kullanılan bileşenler:

- `HfApi`
- `hf_hub_download`
- `list_repo_tree`
- `list_repo_files`
- `get_paths_info`

Sistem, veri deposunun dosya yapısını API üzerinden keşfeder. Beklenen dosya yolları bulunamadığında alternatif keşif ve yerel önbellek mekanizmalarını deneyebilir.

### 5.3 Apache Arrow & Parquet Processing

Veri okuma ve doğrulama altyapısında Apache Arrow ekosisteminin PyArrow bileşenleri kullanılır.

Parquet dosyalarının:

- Metadata bilgileri.
- Satır sayıları.
- Row group sayıları.
- Sütun adları.
- Metin sütunları.
- Dil ve kaynak etiketleri.

incelenir.

Sistem, olası metin sütunlarını farklı sütun adları üzerinden keşfedebilir. Böylece veri kümelerinin farklı şemalarıyla çalışmak kolaylaşır.

### 5.4 Streaming Data Pipeline

Büyük veri kümelerini tamamen RAM'e yüklemek yerine streaming ve batch tabanlı işleme yaklaşımı kullanılır.

Yapılandırma:

```python
PARQUET_BATCH_SIZE = 8192
DOWNLOAD_WORKERS = 8
DOWNLOAD_RETRIES = 4
```

Bu yaklaşımın hedefleri:

- Büyük veri kümelerini kontrollü okumak.
- Bellek tüketimini sınırlamak.
- Kaynaklar arasında dengeli veri akışı sağlamak.
- İndirme hatalarını yeniden denemek.
- İşlem sırasında ilerlemeyi takip etmek.

### 5.5 Dataset Quality & Validation

Veri hazırlama aşamasında aşağıdaki kontroller uygulanır:

- Parquet dosyası bütünlüğü.
- Dosya boyutu doğrulaması.
- Metadata ve şema kontrolü.
- Metin sütunu tespiti.
- Dil etiketi incelemesi.
- Boş veya geçersiz kayıtların ele alınması.
- İsteğe bağlı birebir tekrar kontrolü.
- Örnekleme ve holdout ayırma.

Tekrar tespiti etkinleştirildiğinde Bloom Filter kullanılır.

Bu yapı bellek açısından verimli bir birebir tekrar kontrolü yaklaşımıdır; benzer fakat aynı olmayan metinleri yakalayan bir near-duplicate sistemi değildir.

### 5.6 Reproducibility

Veri örnekleme ve işleme aşamalarında sabit bir random seed kullanılır.

```python
RANDOM_SEED = 20261116
```

Morfoloji örneklemesinde Reservoir Sampling (Algorithm R) yaklaşımı kullanılır.

Bu yöntem, büyük veri akışlarından sınırlı sayıda örnek seçerken tüm veri kümesini bellekte tutma ihtiyacını azaltır.

---

## 6. Performance-Oriented Engineering

RuzgarTokenizer'ın altyapısı yalnızca token üretimine değil, uzun süren eğitim ve veri işleme görevlerinin izlenebilirliğine de odaklanır.

### 6.1 Parallel Processing

Projede farklı görevler için farklı paralellik yaklaşımları kullanılır:

- Rust/Rayon tabanlı tokenizer işlemleri.
- ThreadPoolExecutor tabanlı paralel dosya indirme.
- Zeyrek için çoklu worker süreçleri.
- Python multiprocessing.
- Morfoloji ve veri işleme aşamalarında kontrollü görev dağıtımı.

### 6.2 Memory-Aware Execution

`psutil` üzerinden sistem ve süreç kaynakları izlenir.

İzlenen bilgiler:

- CPU kullanımı.
- Süreç RSS bellek tüketimi.
- Kullanılabilir RAM.
- Toplam RAM.
- Disk boş alanı.
- Disk kullanımı.

Ayrıca cgroup bellek sınırlarını tespit etmeye yönelik kontrol bulunur. Bu, özellikle konteyner tabanlı ortamlarda host RAM miktarının yanıltıcı olmasını önlemeye yardımcı olur.

### 6.3 Fault Tolerance & Recovery

Uzun süreli işlemlerde tek bir modülün başarısız olması bütün projenin otomatik olarak durması anlamına gelmez.

Modül izolasyonu sayesinde aşamalar ayrı ayrı çalıştırılır ve durumları kaydedilir.

Ek mekanizmalar:

- Retry ve exponential backoff.
- Atomic file writes.
- JSONL geçici dosya kullanımı.
- Dosya doğrulama önbelleği.
- Aşama bazlı durum takibi.
- Heartbeat sistemi.
- Hata logları.
- Önceki geçerli çıktıları yeniden kullanma.
- Gerekli aşamaları zorla yeniden çalıştırma seçenekleri.

Atomic write yaklaşımı, bir dosya yazımı tamamlanmadan önce hedef dosyanın yarım kalmış yeni bir sürümle değiştirilmesini önlemeye yardımcı olur.

### 6.4 Persistent Outputs

Proje çıktıları farklı amaçlara göre ayrılmış dizinlerde saklanır:

```text
RUZGAR_V13/
├── datasets/
├── models/
│   └── ruzgar_256k/
├── morphology/
├── logs/
├── benchmarks/
├── export/
└── _tmp/
```

Geçici dosyalar, önbellekler ve ham veri kümeleri model dağıtım paketine dahil edilmez.

---

## 7. Benchmark & Comparative Evaluation

RuzgarTokenizer'ın performansı, aynı metin koleksiyonu üzerinde farklı tokenizer sistemleriyle karşılaştırılmıştır.

Aşağıdaki sonuçlar yüklenen `benchmark_results(1).csv` dosyasından alınmıştır.

### 7.1 Benchmark Dataset

| Metric | Value |
|---|---:|
| Total documents | 51,140 |
| Failed documents | 0 |
| UTF-8 input size | 200 MiB |
| Bytes | 209,715,200 |
| Characters | 192,131,260 |
| Whitespace-separated words | 24,612,007 |

Karşılaştırılan tokenizer'lar aynı belge sayısı, karakter sayısı, byte miktarı ve kelime sayısı üzerinde değerlendirilmiştir.

### 7.2 Token Efficiency Results

| Tokenizer | Vocabulary | Total tokens | Tokens / byte | Tokens / 1,000 words |
|---|---:|---:|---:|---:|
| **Ruzgar-256K** | 256,000 | **35,741,100** | **0.1704** | **1,452.18** |
| OpenAI o200k_base | 200,019 | 57,460,095 | 0.2740 | 2,334.64 |
| Qwen 2.5 | 151,643 | 66,276,713 | 0.3160 | 2,692.86 |
| OpenAI cl100k_base | 100,277 | 75,408,571 | 0.3596 | 3,063.89 |
| DeepSeek V3 | 128,000 | 79,031,654 | 0.3769 | 3,211.10 |
| NEDO 32K | 32,768 | 81,762,570 | 0.3899 | 3,322.06 |

<small>Source: Uploaded comparative benchmark CSV. Vocabulary values are the values recorded in that benchmark.</small>

### 7.3 Token Count Reduction

RuzgarTokenizer'ın ölçülen token sayısı ile diğer tokenizer'ların ölçülen token sayıları arasındaki fark:

| Compared tokenizer | Ruzgar token reduction |
|---|---:|
| NEDO 32K | **56.29%** |
| DeepSeek V3 | **54.77%** |
| OpenAI cl100k_base | **52.61%** |
| Qwen 2.5 | **46.08%** |
| OpenAI o200k_base | **37.80%** |

Bu oranlar, aynı benchmark metinlerinde üretilen toplam token sayılarının karşılaştırılmasından hesaplanmıştır.

Bu sonuçlar, ölçülen veri kümesinde RuzgarTokenizer'ın daha az token ürettiğini gösterir. Tek başına bütün dillerde veya bütün kullanım senaryolarında üstünlük kanıtı değildir.

### 7.4 Throughput Results

| Tokenizer | Documents / sec | MB / sec | Tokens / sec |
|---|---:|---:|---:|
| Ruzgar-256K | 667.13 | 2.61 | 466,251 |
| NEDO 32K | 346.69 | 1.36 | 554,285 |
| OpenAI o200k_base | 3,630.37 | 14.20 | 4,079,031 |
| OpenAI cl100k_base | 2,910.57 | 11.38 | 4,291,790 |
| Qwen 2.5 | 692.62 | 2.71 | 897,627 |
| DeepSeek V3 | 706.62 | 2.76 | 1,092,013 |

RuzgarTokenizer'ın token sayısı verimliliği ile encoding hızı birbirinden ayrı değerlendirilmelidir.

Benchmark dosyasına göre OpenAI encodings daha yüksek throughput değerleri göstermiştir. Dolayısıyla RuzgarTokenizer'ın temel iddiası bu ölçümde en yüksek encoding hızına sahip olmak değil, Türkçe benchmark metninde token sayısını azaltmaktır.

### 7.5 Latency Results

| Tokenizer | P50 latency | P95 latency | P99 latency |
|---|---:|---:|---:|
| Ruzgar-256K | 0.662 ms | 4.634 ms | 11.256 ms |
| NEDO 32K | 1.266 ms | 9.160 ms | 22.525 ms |
| OpenAI o200k_base | 0.120 ms | 0.768 ms | 1.780 ms |
| OpenAI cl100k_base | 0.151 ms | 0.994 ms | 2.303 ms |
| Qwen 2.5 | 0.653 ms | 4.447 ms | 10.422 ms |
| DeepSeek V3 | 0.626 ms | 4.430 ms | 10.363 ms |

Latency değerleri milisaniye cinsindendir.

P50 medyan gecikmeyi, P95 ölçümlerin yüzde 95'inin altında kaldığı gecikmeyi, P99 ise yüzde 99'luk gecikme sınırını ifade eder.

### 7.6 Benchmark Interpretation

Benchmark sonuçları üç farklı performans boyutunu gösterir:

**1. Token efficiency**

RuzgarTokenizer, bu 200 MiB Türkçe benchmark setinde karşılaştırılan sistemlerden daha az token üretmiştir.

**2. Encoding throughput**

Token üretim hızı bakımından sonuçlar tokenizer uygulamasına ve ölçüm ortamına göre değişmektedir. CSV'de OpenAI encodings daha yüksek throughput göstermiştir.

**3. Latency**

RuzgarTokenizer'ın P50/P95/P99 değerleri düşük milisaniye seviyesindedir; ancak aynı ölçümde OpenAI encodings daha düşük latency değerlerine sahiptir.

Benchmark sonuçlarını yeniden üretilebilir kılmak için kullanılan donanım, yazılım sürümleri, tokenizer wrapper'ları, batch büyüklükleri, warm-up sayısı ve ölçüm yöntemiyle birlikte ham CSV'nin yayımlanması önerilir. Mevcut CSV bu bilgilerin tamamını içermemektedir.

---

## 8. Internal Benchmarking & Validation

Harici karşılaştırma CSV'sinden ayrı olarak V13.2 kodunda kendi benchmark aşaması da bulunur.

Bu aşama şu metrikleri üretir:

- Toplam token sayısı.
- Token per word.
- Characters per token.
- Bytes per token.
- Tokens per 1,000 characters.
- Batch documents per second.
- Single-thread documents per second.
- Batch tokens per second.
- CPU thread sayısı.
- Round-trip doğruluğu.
- Tokenizer dosya boyutu.
- Eğitim süresi.
- Eğitim veri miktarı.

### Round-Trip Validation

Tokenizer'ın encoding ve decoding işlemlerinin metni koruyup korumadığı kontrol edilir.

```python
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded.ids)
```

Testin amacı, özel token ayarları dışında giriş metninin decoding sonrasında aynı kalıp kalmadığını doğrulamaktır.

### Word Fertility

Word fertility analizi, kelimelerin ortalama kaç token ile temsil edildiğini inceler.

Üretilen metrikler:

- Kelime türü başına ortalama token.
- Frekans ağırlıklı token/kelime oranı.
- Tek token ile temsil edilen kelimelerin yüzdesi.

Bu ölçüm, Türkçedeki sık kullanılan kelimelerin tokenization davranışını incelemek için kullanılır.

---

## 9. Twelve-Stage Development Pipeline

V13.2 projesinin ana akışı modüler aşamalardan oluşur.

| Stage | Component | Purpose |
|---:|---|---|
| 1 | Environment & packages | Paketlerin ve çalışma ortamının hazırlanması |
| 2 | Dataset discovery | Veri kaynaklarının ve Parquet dosyalarının keşfi |
| 3 | Dataset download | Dosyaların indirilmesi ve mevcut dosyaların korunması |
| 4 | Parquet validation | Şema, metadata ve metin sütunu doğrulaması |
| 5 | Streaming data preparation | Veri akışı, filtreleme ve örnekleme |
| 6 | Corpus statistics | Veri ve kelime istatistiklerinin oluşturulması |
| 7 | Word corpus | Morfoloji için kelime korpusunun hazırlanması |
| 8 | Zeyrek analysis | Türkçe morfolojik analiz |
| 9 | Morfessor training | Denetimsiz morfem keşfi |
| 10 | Unified morphology | Morfoloji sonuçlarının birleştirilmesi |
| 11 | Benchmark & validation | Tokenizer performansının ölçülmesi |
| 12 | ZIP export | Model ve araştırma çıktılarının paketlenmesi |

Bu yapı, her aşamanın ayrı durum ve hata bilgileriyle takip edilmesini sağlar.

---

## 10. Technology Stack

| Technology | Role |
|---|---|
| Python | Pipeline orchestration and data processing |
| Rust | High-performance tokenizer engine |
| Rayon | Parallel execution in Rust tokenizer operations |
| Hugging Face Tokenizers | ByteLevel BPE training and encoding |
| Hugging Face Hub | Dataset discovery and downloads |
| PyArrow | Apache Arrow and Parquet processing |
| Zeyrek | Turkish morphological analysis |
| Morfessor | Unsupervised morpheme discovery |
| SQLite | Disk-backed morphology indexing |
| JSON / JSONL | Structured experiment outputs |
| psutil | CPU, memory and process monitoring |
| multiprocessing | Parallel morphology workloads |
| ThreadPoolExecutor | Concurrent downloads |
| hashlib | File and input hashing |
| zipfile | Model and research artifact packaging |
| Marimo | Notebook-based execution environment |

---

## 11. Reliability, Logging & Reproducibility

Uzun süreli veri ve eğitim işlemlerinde hata yönetimi önemli bir tasarım unsurudur.

RuzgarTokenizer'ın V13.2 altyapısında:

- Her modülün çalışma durumu tutulur.
- Modül çalışma süreleri kaydedilir.
- Hata mesajları ve traceback bilgileri loglanır.
- Uzun işlemler heartbeat mesajları üretir.
- İndirme hataları tekrar denenebilir.
- Geçerli Parquet dosyaları doğrulama önbelleğinde tutulur.
- JSON ve JSONL dosyaları atomik biçimde yazılır.
- Önceki geçerli eğitim ve analiz çıktıları yeniden kullanılabilir.
- Yeniden eğitim ve yeniden örnekleme ayrı parametrelerle zorlanabilir.

Bu yaklaşım, büyük veri işleme görevlerinde hata sonrası toparlanmayı ve deneylerin izlenebilirliğini kolaylaştırır.

---

## 12. Model Packaging & Distribution

Proje, tokenizer ve araştırma çıktılarının paylaşılabilmesi için ZIP paketleme aşamasına sahiptir.

Paketleme sistemi:

- Model dosyalarını toplar.
- Morfoloji sonuçlarını dahil eder.
- Benchmark raporlarını ekler.
- Log dosyalarını ekler.
- README dosyasını dahil eder.
- Ham Parquet datasetlerini dışarıda bırakır.
- Geçici dosyaları ve önbellekleri dışarıda bırakır.
- ZIP dosyasını oluşturduktan sonra bütünlük kontrolü yapar.

GitHub'ın dosya boyutu sınırları ve büyük model dosyalarının yönetimi nedeniyle büyük dağıtım dosyaları ZIP paketleri ve Hugging Face Hub üzerinden paylaşılabilir.

**Hugging Face:**  
https://huggingface.co/brun12/RuzgarTokenizer

---

## 13. Installation & Usage

Tokenizer dosyasını Hugging Face deposundan indirebilirsiniz.

Gerekli paket:

```bash
pip install tokenizers huggingface_hub
```

Python örneği:

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tokenizer_path = hf_hub_download(
    repo_id="brun12/RuzgarTokenizer",
    filename="models/ruzgar_256k/tokenizer.json"
)

tokenizer = Tokenizer.from_file(tokenizer_path)

text = "RuzgarTokenizer Türkçe metinleri tokenlara ayırır."

encoded = tokenizer.encode(text)

print("Token IDs:", encoded.ids)
print("Tokens:", encoded.tokens)
print("Token count:", len(encoded.ids))

decoded = tokenizer.decode(encoded.ids)
print("Decoded:", decoded)
```

---

## 14. Project Goals & Future Research

RuzgarTokenizer'ın geliştirme hedefleri:

- Türkçe token verimliliğini daha geniş veri kümelerinde incelemek.
- Morfolojik sınır uyumunu daha kapsamlı değerlendirmek.
- Farklı pre-tokenization yaklaşımlarını karşılaştırmak.
- Türkçe kelime fertility analizlerini genişletmek.
- Daha fazla bağımsız benchmark veri kümesi oluşturmak.
- Encoding ve decoding performansını optimize etmek.
- Farklı vocabulary boyutlarının sonuçlarını incelemek.
- Bağımsız holdout veri kümelerinde genelleme ölçümlerini artırmak.
- Benchmark ortamını ve ölçüm protokolünü daha ayrıntılı yayımlamak.

---

## 15. Developer

**RuzgarTokenizer**  
Developed by **Rüzgar Vatansever**

Project repository:  
https://huggingface.co/brun12/RuzgarTokenizer

---

# ENGLISH DOCUMENTATION

## 1. Overview

**RuzgarTokenizer** is a Turkish-focused tokenization and morphological research project built around a ByteLevel Byte Pair Encoding (BPE) architecture with a target vocabulary size of 256,000 tokens.

The project aims to investigate token efficiency, text segmentation, morphological structure, and encoding performance in Turkish-language corpora.

Rather than treating tokenization as an isolated operation, RuzgarTokenizer combines a high-performance tokenizer engine with a large-scale data processing pipeline, Turkish morphological analysis, unsupervised morpheme discovery, and comparative benchmarking.

Its development references include NEDO and Morpheus, while its comparative evaluation includes NEDO, OpenAI public encodings, Qwen 2.5, and DeepSeek V3.

## 2. Core Tokenizer Architecture

### ByteLevel BPE

The core algorithm is Byte-Level Byte Pair Encoding.

BPE learns frequent symbol combinations from a training corpus and represents text using a vocabulary of learned subword units.

The byte-level design provides a flexible representation for Unicode text and supports Turkish characters such as:

`ç, ğ, ı, İ, ö, ş, ü`

The configured target vocabulary is 256,000 entries.

| Configuration | Value |
|---|---:|
| Target vocabulary | 256,000 |
| Algorithm | ByteLevel BPE |
| Minimum BPE frequency | 2 |
| Special tokens | `<|endoftext|>`, `<|pad|>` |
| Default pre-tokenizer | `turkish_split` |

### Rust-Based Execution Engine

RuzgarTokenizer uses Hugging Face Tokenizers for its BPE training and tokenization operations.

The architecture combines Python-based pipeline orchestration with a Rust-powered tokenizer engine. Rayon and configurable thread settings support parallel CPU execution.

The Python layer manages data acquisition, preprocessing, morphology, experiment tracking, reporting, and packaging.

The tokenizer engine handles performance-critical tokenization operations.

**The V13.2 implementation uses CPU-based Rust/Rayon execution for BPE training. It does not claim GPU-accelerated BPE training.**

## 3. Turkish-Aware Pre-Tokenization

Turkish is an agglutinative language in which suffixes can create many word forms from a single root.

This creates important tokenization challenges:

- Long inflected word forms.
- Frequent suffix combinations.
- Root and suffix boundary representation.
- Turkish-specific characters.
- Apostrophe handling.
- Vocabulary allocation across word forms.

RuzgarTokenizer provides two configurable pre-tokenization modes:

- `turkish_split`
- `gpt2`

The default configuration is `turkish_split`.

These modes determine how text is segmented before BPE training; they are not alternative BPE algorithms.

## 4. Morphological Research Framework

RuzgarTokenizer includes a dedicated morphology research pipeline designed to examine the relationship between learned BPE boundaries and Turkish word structure.

### Zeyrek

Zeyrek is used for Turkish morphological analysis.

The pipeline records available analysis results, lemmas, ambiguity information, analysis counts, and part-of-speech statistics.

A multiprocessing-oriented execution structure supports large word collections. Results are stored in structured JSONL files.

### Morfessor

Morfessor is used for unsupervised morpheme discovery through its `BaselineModel` implementation.

The pipeline includes:

1. Word corpus loading.
2. Model initialization.
3. Batch training.
4. Epoch and cost tracking.
5. Binary model serialization.
6. Model reload verification.
7. Viterbi segmentation.
8. Structured segmentation output.

The system records average morphemes per word, unique morpheme counts, frequent morphemes, unsplit words, and segmentation errors.

Morfessor is an unsupervised method. Its segmentations are not treated as gold-standard linguistic annotations.

### Unified Morphology

Zeyrek, Morfessor, custom suffix heuristics, and corpus-derived phonological evidence can be combined into a unified morphology dataset.

SQLite provides disk-backed word indexing, reducing the need to load all intermediate records into memory.

### Morphological Boundary Alignment

The project measures how BPE token boundaries overlap with Morfessor segmentation boundaries.

It reports:

- Boundary precision.
- Morfessor boundary recall.
- F1 score.
- Exact boundary match percentage.

These measurements describe segmentation agreement, not definitive linguistic correctness.

## 5. Large-Scale Data Engineering

V13.2 defines three Hugging Face dataset sources:

| Dataset | Repository | Configured weight |
|---|---|---:|
| FineWeb-2 Turkish | `HuggingFaceFW/fineweb-2` | 40% |
| Turkish Corpus 100B | `hasankursun/turkish-corpus-100b` | 40% |
| Umarigan Turkish Corpus | `umarigan/turkish_corpus` | 20% |

These are configured source weights, not a guarantee of the exact final corpus distribution.

The data pipeline uses Hugging Face Hub APIs for repository discovery and downloading, and PyArrow for Parquet inspection and processing.

Its data engineering components include:

- Repository and file discovery.
- Parquet metadata inspection.
- Schema and text-column detection.
- Language-label probing.
- Batch-based reading.
- Streaming document processing.
- Configurable download concurrency.
- Retry and recovery mechanisms.
- Optional Bloom Filter exact-duplicate detection.
- Reservoir sampling.
- Holdout dataset support.
- Dataset validation caching.

The default configuration includes an 8,192-row Parquet batch size, eight download workers, and four download retries.

## 6. Performance-Oriented Infrastructure

RuzgarTokenizer is designed to monitor and manage long-running workloads.

Its infrastructure includes:

- Rust/Rayon tokenizer execution.
- ThreadPoolExecutor-based downloads.
- Multiprocessing-oriented morphology analysis.
- Process and memory monitoring with psutil.
- cgroup memory-limit detection.
- SQLite-backed intermediate indexing.
- Atomic JSON and JSONL output.
- Retry with exponential backoff.
- Heartbeat monitoring.
- Stage-level execution tracking.
- Persistent intermediate results.
- Configurable retraining and resampling.

These mechanisms help make large corpus processing more observable, recoverable, and manageable.

## 7. Comparative Benchmark Results

The supplied benchmark CSV records results from six tokenizers evaluated over the same text collection.

### Evaluation Dataset

| Metric | Value |
|---|---:|
| Documents | 51,140 |
| Failed documents | 0 |
| Input size | 200 MiB |
| UTF-8 bytes | 209,715,200 |
| Characters | 192,131,260 |
| Whitespace-separated words | 24,612,007 |

### Token Efficiency

| Tokenizer | Vocabulary | Tokens | Tokens / byte | Tokens / 1,000 words |
|---|---:|---:|---:|---:|
| **Ruzgar-256K** | 256,000 | **35,741,100** | **0.1704** | **1,452.18** |
| OpenAI o200k_base | 200,019 | 57,460,095 | 0.2740 | 2,334.64 |
| Qwen 2.5 | 151,643 | 66,276,713 | 0.3160 | 2,692.86 |
| OpenAI cl100k_base | 100,277 | 75,408,571 | 0.3596 | 3,063.89 |
| DeepSeek V3 | 128,000 | 79,031,654 | 0.3769 | 3,211.10 |
| NEDO 32K | 32,768 | 81,762,570 | 0.3899 | 3,322.06 |

### Token Count Reduction

| Comparison | Reduction |
|---|---:|
| Against NEDO 32K | **56.29%** |
| Against DeepSeek V3 | **54.77%** |
| Against OpenAI cl100k_base | **52.61%** |
| Against Qwen 2.5 | **46.08%** |
| Against OpenAI o200k_base | **37.80%** |

These percentages compare total token counts on the supplied benchmark dataset. They should not be interpreted as universal improvements across all languages or workloads.

### Throughput

| Tokenizer | Documents/sec | MB/sec | Tokens/sec |
|---|---:|---:|---:|
| Ruzgar-256K | 667.13 | 2.61 | 466,251 |
| NEDO 32K | 346.69 | 1.36 | 554,285 |
| OpenAI o200k_base | 3,630.37 | 14.20 | 4,079,031 |
| OpenAI cl100k_base | 2,910.57 | 11.38 | 4,291,790 |
| Qwen 2.5 | 692.62 | 2.71 | 897,627 |
| DeepSeek V3 | 706.62 | 2.76 | 1,092,013 |

The results distinguish token efficiency from encoding throughput. RuzgarTokenizer produces fewer tokens on this dataset, while the supplied CSV reports higher throughput for the OpenAI encodings.

### Latency

| Tokenizer | P50 | P95 | P99 |
|---|---:|---:|---:|
| Ruzgar-256K | 0.662 ms | 4.634 ms | 11.256 ms |
| NEDO 32K | 1.266 ms | 9.160 ms | 22.525 ms |
| OpenAI o200k_base | 0.120 ms | 0.768 ms | 1.780 ms |
| OpenAI cl100k_base | 0.151 ms | 0.994 ms | 2.303 ms |
| Qwen 2.5 | 0.653 ms | 4.447 ms | 10.422 ms |
| DeepSeek V3 | 0.626 ms | 4.430 ms | 10.363 ms |

The supplied CSV does not include every hardware and software detail needed to establish a fully controlled cross-library benchmark. Results should therefore be interpreted as measurements from the recorded benchmark run.

## 8. Internal Validation

In addition to comparative benchmarking, V13.2 contains an internal benchmark and validation stage.

It measures:

- Total token count.
- Tokens per word.
- Characters per token.
- Bytes per token.
- Tokens per 1,000 characters.
- Batch encoding throughput.
- Single-thread document throughput.
- Round-trip reconstruction.
- Tokenizer file size.
- Training duration.
- Training corpus statistics.

Round-trip validation checks whether encoding and decoding preserve the original input text.

Word fertility analysis measures how efficiently individual word forms are represented.

## 9. Twelve-Stage Pipeline

| Stage | Component | Description |
|---:|---|---|
| 1 | Environment | Package installation and environment setup |
| 2 | Discovery | Dataset repository and file discovery |
| 3 | Download | Concurrent Parquet downloads |
| 4 | Validation | Dataset schema and metadata validation |
| 5 | Streaming | Document processing and filtering |
| 6 | Statistics | Corpus and sampling statistics |
| 7 | Word corpus | Morphology word corpus preparation |
| 8 | Zeyrek | Turkish morphological analysis |
| 9 | Morfessor | Unsupervised morpheme discovery |
| 10 | Unified morphology | Morphology output integration |
| 11 | Benchmark | Tokenization performance evaluation |
| 12 | Packaging | Verified ZIP export |

## 10. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Pipeline orchestration |
| Rust | Tokenizer execution engine |
| Rayon | Parallel CPU execution |
| Hugging Face Tokenizers | ByteLevel BPE |
| Hugging Face Hub | Dataset discovery and downloads |
| PyArrow | Parquet processing |
| Zeyrek | Turkish morphology |
| Morfessor | Unsupervised segmentation |
| SQLite | Intermediate indexing |
| psutil | Resource monitoring |
| multiprocessing | Parallel processing |
| ThreadPoolExecutor | Concurrent downloads |
| JSON / JSONL | Structured outputs |
| hashlib | Hashing |
| zipfile | Artifact packaging |
| Marimo | Notebook execution |

## 11. Packaging & Distribution

The project includes an automated ZIP packaging stage.

The package can contain:

- Tokenizer model files.
- Morphology outputs.
- Benchmark reports.
- Logs.
- README documentation.

Raw Parquet datasets, temporary files, caches, and intermediate SQLite files are excluded from the distribution package.

Large project artifacts can be distributed through ZIP archives and Hugging Face Hub.

**Hugging Face Repository:**  
https://huggingface.co/brun12/RuzgarTokenizer

## 12. Installation & Usage

Install the required packages:

```bash
pip install tokenizers huggingface_hub
```

Load the tokenizer:

```python
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

path = hf_hub_download(
    repo_id="brun12/RuzgarTokenizer",
    filename="models/ruzgar_256k/tokenizer.json"
)

tokenizer = Tokenizer.from_file(path)

text = "RuzgarTokenizer is a Turkish-focused tokenizer."

encoded = tokenizer.encode(text)

print(encoded.ids)
print(encoded.tokens)
print(len(encoded.ids))

print(tokenizer.decode(encoded.ids))
```

## 13. Future Research

Planned research directions include:

- Broader Turkish corpus evaluation.
- More independent holdout benchmarks.
- Expanded morphological alignment analysis.
- Vocabulary-size experiments.
- Additional pre-tokenization comparisons.
- Improved benchmark reproducibility.
- Encoding and decoding optimization.
- More extensive word fertility studies.

## 14. Author

**RuzgarTokenizer**  
Created and developed by **Rüzgar Vatansever**.

Repository: https://huggingface.co/brun12/RuzgarTokenizer

---

*Benchmark figures in this document are taken from the supplied CSV. Implementation details are based on the supplied V13.2 source code. Configured parameters, measured results, and research interpretations are intentionally distinguished.*
