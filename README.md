# Laporan Proyek Machine Learning - Salsa Tashfiyatul Qolbi

## Domain Proyek
....

## Business Understanding

### Problem Statements
* Bagaimana cara merekomendasikan produk yang relevan kepada pengguna berdasarkan produk yang mereka cari?

### Goals
* Membangun sistem rekomendasi untuk produk Sephora dan menyediakan rekomendasi top-N produk serupa berdasarkan produk dan kategori produk.

### Solution Statements
* Sistem akan mengekstrak fitur penting dari setiap produk, seperti bahan (ingredients), deskripsi produk, kategori, dan jenis produk.
* Dengan menggunakan teknik TF-IDF vectorization dan cosine similarity, sistem akan menghitung kemiripan antar produk berdasarkan kontennya.
* Sistem ini tidak memerlukan data dari pengguna lain (seperti pada collaborative filtering), sehingga bisa langsung memberikan rekomendasi bahkan untuk pengguna baru.

## Data Understanding
Dataset Sephora Products and Skincare Reviews yang berasal dari [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?select=product_info.csv). Dataset ini memiliki 8494 baris dan 27 kolom yang terdiri atas 15 kolom numerik dan 12 kolom kategorikal. Penjelasan lebih rinci akan dijelaskan dalam tahap berikut ini:

**Informasi Dataset**
# Sephora Product & Skincare Dataset
| **Judul**       | Life Expectancy Dataset                                                             |                  
|-----------------|-------------------------------------------------------------------------------------|
| **Author**      | Nady Inky                                                                           |
| **Source**      | [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews?select=product_info.csv)          |
| **Visibility**  | Public                                                                              |
| **Usability**   | 10.00                                                                               |

**Metadata**
| Kolom                         | Deskripsi                                         | Tipe Data    |
|-------------------------------|---------------------------------------------------|--------------|
| Product_id                    | kode unik produk dari situs                       | Kategorikal  |
| Product_name                  | nama lengkap dari produk                          | Kategorikal  |
| Brand_id                      | kode unik untuk merek produk dari situs           | Integer      |
| Brand_name                    | nama lengkap dari merek produk                    | Kategorikal  |
| Loves_count                   | jumlah orang yang menandai produk sebagai favorit | Integer      |
| Rating                        | rata-rata ulasan produk berdasarkan ulasan pengguna | Float      |
| Reviews                       | jumlah ulasan pengguna untuk sebuah produk        | Float        |
| Size                          | ukuran dari produk, dapat dalam bentuk oz, ml, g, packs, atau satuan lain sesuai dengan tipe produk | Kategorikal |
| Variation_type                | tipe dari parameter variasi untuk produk (contoh: ukuran, warna) | Kategorikal |
| Variation_value               | nilai spesifik dari parameter variasi produk (contoh: 100 mL, Golden Sand) | Kategorikal |
| Variation_desc                | deskripsi dari parameter variasi produk (contoh: tone untuk kulit pucat) | Kategorikal |
| Ingredients                   | list komposisi yang terkandung dalam produk, contoh [‘Product variation 1:’, ‘Water, Glycerin’, ‘Product variation 2:’, ‘Talc, Mica’] or if no variations [‘Water, Glycerin’] | Kategorikal |
| Price_usd                     | harga produk dalam dollar US | Float |
| value_price_usd | potensi penghematan biaya produk, ditampilkan disebelah harga asli | Float |
| Sale_price_usd | harga promo produk dalam dollar US | Float |
| Limited_edition | indikasi produk adalah edisi terbatas atau tidak (1-iya, 0-tidak) | Integer |
| New | indikasi produk barang baru atau bukan (1-iya, 0-tidak) | Integer |
| Online_only | indikasi produk hanya dijual secara online atau tidak (1-iya, 0-tidak) | Integer |
| Out_of_stock | indikasi produk sedang kehabisan stok atau tidak (1-iya, 0-tidak) | Integer |
| Sephora_exclusive | indikasi produk adalah barang ekslusif Sephora atau bukan (1-iya, 0-tidak) | Integer |
| Highlights | list berisi tag atau fitur yang menonjolkan atribut produk (contoh: vegan, matte finish) | Kategorikal |
| Primary_category | kategori pertama dibagian breadcrumb | Kategorikal |
| Secondly_category | kategori kedua dibagian breadcrumb | Kategorikal |
| Tertiary_category | kategori ketiga dibagian breadcrumb | Kategorikal |
| Child_count | jumlah variasi dari ketersediaan produk | Integer |
| Child_mac_price | harga tertinggi dari variasi produk | Float |
| Child_min_price | harga terendah dari variasi produk | Float |

### Exploratory Data Analysis
Pada tahap ini dilakukan analisis untuk data yang ada di dalam dataset seperti dilihat
| Kolom                             | Jumlah Non-NUll    | TIpe Data     |
|-----------------------------------|--------------------|---------------|
|   product_id     |     8494 non-null  | object  |
|  product_name    |    8494 non-null  | object |
|  brand_id        |   8494 non-null   |int64  |
|  brand_name      |    8494 non-null   |object |
|  loves_count     |    8494 non-null |  int64  |
|   rating         |     8216 non-null|   float64|
|  reviews         |    8216 non-null |  float64|
|  size            |    6863 non-null |  object |
|  variation_type  |    7050 non-null |  object |
|  variation_value |    6896 non-null |  object |
|  variation_desc  |    1250 non-null |  object |
| ingredients      |   7549 non-null  | object |
| price_usd        |   8494 non-null  | float64|
| value_price_usd  |   451 non-null   | float64|
|  sale_price_usd  |    270 non-null  |  float64|
| limited_edition  |   8494 non-null  | int64  |
| new              |   8494 non-null  | int64  |
| online_only      |   8494 non-null  | int64  |
| out_of_stock     |   8494 non-null  | int64  |
| sephora_exclusive|   8494 non-null  | int64  |
| highlights       |   6287 non-null  | object |
| primary_category |   8494 non-null  | object |
| secondary_category|  8486 non-null  | object |
| tertiary_category |  7504 non-null  | object |
| child_count       |  8494 non-null  | int64  |
| child_max_price   |  2754 non-null  | float64|
| child_min_price   |  2754 non-null  | float64|

Output di atas menunjukkan bahwa dataset memiliki 8494 data dan 27 kolom.
- Terdapat 12 tipe data object
- Terdapat 8 tipe data int64
- Terdapat 7 tipe data float64
- Terdapat Missing Value

### EDA - Univariate Analysis
Berikut adalah persebaran data atau distribusi data dari fitur numerik.
... 

Berikut adalah persebaran data atau distribusi data dari fitur kategorikal.
...

### Correlation Matrix
...
Terlihat pada metrik korelasi bahwa setiap fitur numerik tidak memiliki korelasi satu sama lain, kecuali fitur reviews dan loves_count. Kedua fitur tersebut berkorelasi positif, yang artinya jika fitur reviews meningkat maka fitur loves_count juga ikut meningkat.

## Data Preparation
Teknik yang akan dilakukan:
- Missing value : cek missing value dan penanganannya
- Duplikasi data : cek apakah terdapat data yang terduplikasi
- Fitur selection : menghapus beberapa fitur yang tidak relevan
- Fitur Combine : menggabungkan beberapa fitur menjadi satu fitur
- Text Preprocessing : membersihkan data dari simbol, tanda baca, atau kata yang tidak terlalu bermakna

### Missing Value
Walaupun penanganan missing value termasuk tahap data preparation, namun tetap dilakukan sebelum EDA agar hasil dari EDA dapat memberikan hasil yang maksimal. Terlihat bahwa dalam dataset terdapat missing value. Berikut penanganan yang sesuai dengan tipe data masing-masing fitur:
- Untuk fitur rating dan reviews menggunakan imputasi dengan median. Kkarena fitur ini termasuk fitur numerikal, maka imputasi yang dilakukan menggunakan median.
- Untuk fitur size, variation_type, variation_value, ingredients, highlights, secondary_category, tertiary_category imputasi dengan string "Unknown". Karena fitur tersebut termasuk fitur kategorikal, maka imputasi yang dilakukan menggunakan string.
- Untuk variation_desc, value_price_usd, sale_price_usd, child_min_price, child_max_price akan dihapus karena terlalu banyak missing value.

### Duplikasi Data
Tidak terdapat data yang terduplikasi. Sama hal nya dengan missing value, cek duplikasi data dilakukan sebelum EDA.

### Fitur Selection
Dilakukan untuk memilih fitur-fitur yang paling relevan dengan sistem rekomendasi, tujuannya untuk meningkatkan akurasi dan generalisasi model. Kenapa banyak fitur yang dihapus? Karena sistem rekomendasi ini sangat sederhana dan berbasis teks. Fitur yang dipilih pun akan cocok dengan pendekatan Content-Based Filtering dengan menggunakan TF-IDF dan Cosine Similarity.

### Fitur Combine
Dilakukan untuk membuat fitur baru dari kombinasi fitur lama untuk memperkuat sinyal dalam data. Tujuannya untuk menyediakan informasi tambahan yang tidak langsung terlihat dari fitur asli dan menangkap interaksi antar fitur, Ini akan memudahkan model dalam mengidentifikasi dua produk yang serupa, tapi beda brand.

### Text Preprocessing
Tahap ini dilakukan agar data yang ada bersih dari karakter, simbol, atau kata yang tidak bermakna dan saat diaplikasikan ke model hasilnya dapat maksimal.

## Data Modelling
Hal yang dilakukan:
- TF-IDF vektorizer
- Cosine Similarity
- Pengujian Model

### TF-IDF vektorizer
TF-IDF adalah metode konversi teks menjadi angka yang mencerminkan pentingnya suatu kata dalam sebuah dokumen relatif terhadap seluruh korpus
Kelebihan:
- Sederhana dan cepat digunakan
- Mengurangi bobot kata umum
- Fokus pada kata yang relevan
Kekurangan:
- Tidak memahami konteks
- Tidak menangkap urutan kata
- Tidak memperhitungkan sinonim

### Cosine Similarity
Cosine similarity adalah ukuran kesamaan antara dua vektor berdasarkan sudut kosinus di antara mereka, bukan nilai absolut atau jarak euclidean. Fungsi utamanya untuk mengukur kemiripan dokumen teks, terutama yang direpresentasikan dalam bentuk TF-IDF.
Kelebihan:
- Tidak dipengaruhi oleh panjang dokumen
- Cepat dihitung dan efisien
Kekurangan:
- Tidak mempertimbangkan frekuensi kata yang tinggi
- Tidak memahami makna kata

### Pengujian Model
...

## Evaluasi

### Precision
Untuk Content-Based Filtering digunakan metrik evaluasi precision untuk melihat seberapa baik sistem rekomendasi menggunakan Cosine Similarity. Precision adalah metrik evaluasi yang akan mengukur seberapa banyak dari hasil yang ditampilkan oleh sistem benar-benar relevan. Precision memberikan gambaran seberapa banyak dari prediksi yang dikategorikan positif benar-benar merupakan kasus positif yang sebenarnya.

![Screenshot 2025-05-29 010903.png]()

Dimana:
- TP (True Positive), jumlah kejadian positif yang diprediksi dengan benar.
- FP (False Positive), jumlah kejadian positif yang diprediksi dengan salah.

Karena dalam pengujian dilakukan dua kali, maka untuk menghitung precision akan menggunakan average precision. Average precision dapat memberikan informasi terkait rata-rata performa sistem rekomendasi.

![image.png]()

Dimana:
- n : jumlah pengujian
- Precision@10𝑖: nilai precision dari setiap pengujian.

Pada pengujian pertama:
recommend('Cleanser', category='skincare') menghasilkan 7 rekomendasi, dimana ada 2 rekomendasi yang tidak relevan. Artinya, pengujian pertama memiliki precision sebesar **0.7**.

Pada pengujian kedua:
recommend('Waterproof', category='makeup') menghasilkan 10 rekomendasi, dimana ada 1 rekomendasi yang tidak relevan. Artinya, pengujian pertama memiliki precision sebesar **0.9**.

Rata-rata precision:
Pada pengujian pertama didapat hasil 0.7 dan pada pengujian kedua didapat hasil 0.9, maka rata-rata precision adalah sebesar **0.8**.
