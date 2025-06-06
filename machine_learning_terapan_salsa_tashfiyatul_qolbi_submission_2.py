# -*- coding: utf-8 -*-
"""Copy of Machine_Learning_Terapan_Salsa_Tashfiyatul_Qolbi_Submission_1

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vT7WQvXdpCAUgsuqa5PNXwLTTrNucNPD

# Model Sistem Rekomendasi Produk Skincare : Content-Based Filtering

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

## Data Understanding

Dalam tahap ini, data akan diproses untuk memahami isi dari dataset.

### Data Loading

#### Import Library

Mengimport seluruh library yang dibutuhkan untuk analisis data dan pembangunan model machine learning.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import kagglehub
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate

"""Fungsi semua library yang diimport:
1. **import pandas as pd** : untuk manipulasi dan analisis data
2. **import numpy as np** : untuk operasi numerik dan array multidimensi
3. **import matplotlib.pyplot as plt** : untuk visualisasi data
4. **import seaborn as sns** : untuk visualisasi statistik
5. **from sklearn.impute import SimpleImputer** : untuk mengimputasi nilai yang hilang
6. **import kagglehub** : untuk mengakses dataset dari KaggleHub
7. **import nltk** : library NLP untuk mengolah teks
8. **nltk.download('stopwords')** : mendowload resource stopwords
9. **from nltk.corpus import stopwords** : untuk mengenali kata-kata umum yang tidak terlalu bermakna
10. **import re** :  untuk membersihkan teks
11. **from sklearn.feature_extraction.text import TfidfVectorizer** : untuk mengubah teks menjadi angka berbasis bobot TF-IDF untuk mengukur pentingnya kata
12. **from sklearn.metrics.pairwise import cosine_similarity** : untuk mengukur seberapa mirip dua teks berdasarkan sudut vektor dalam ruang multidimensi
13. **from tabulate import tabulate** : untuk membuat tabel dalam format teks yang rapi dan mudah dibaca dari data seperti list, dictionary, atau DataFrame.

#### Memuat Dataset

Pada tahap ini memuat dataset ke dalam notebook. Karena dataset memiliki format CSV, maka menggunakan library pandas untuk membacanya.
"""

path = kagglehub.dataset_download("nadyinky/sephora-products-and-skincare-reviews")
file_path = path + "/product_info.csv"
data = pd.read_csv(file_path)

"""### Exploratory Data Analysis (EDA)

Analisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

#### Cek Tipe Data

Dilakukan untuk menampilkan jumlah baris, kolom, tipe data, dan jumlah nilai non-null.
"""

data.info()

"""Terlihat bahwa dataset memiliki 8494 baris dan 27 kolom yang terdiri dari 12 kolom data kategorikal dan 15 kolom data numerik. Berikut adalah seluruh fitur yang ada dalam dataset:
1. Product_id : kode unik produk dari situs
2. Product_name : nama lengkap dari produk
3. Brand_id : kode unik untuk merek produk dari situr
4. Brand_name : nama lengkap dari merek produk
5. Loves_count : jumlah orang yang menandai produk sebagai favorit
6. Rating : rata-rata ulasan produk berdasarkan ulasan pengguna
7. Reviews : jumlah ulasan pengguna untuk sebuah produk
8. Size : ukuran dari produk, dapat dalam bentuk oz, ml, g, packs, atau satuan lain sesuai dengan tipe produk
9. Variation_type : tipe dari parameter variasi untuk produk (contoh: ukuran, warna)
10. Variation_value : nilai spesifik dari parameter variasi produk (contoh: 100 mL, Golden Sand)
11. Variation_desc : deskripsi dari parameter variasi produk (contoh: tone untuk kulit pucat)
12. Ingredients : list komposisi yang terkandung dalam produk, contoh [‘Product variation 1:’, ‘Water, Glycerin’, ‘Product variation 2:’, ‘Talc, Mica’] or if no variations [‘Water, Glycerin’]
13. Price_usd : harga produk dalam dollar US
14. value_price_usd : potensi penghematan biaya produk, ditampilkan disebelah harga asli
15. Sale_price_usd : harga promo produk dalam dollar US
16. Limited_edition : indikasi produk adalah edisi terbatas atau tidak (1-iya, 0-tidak)
17. New : indikasi produk barang baru atau bukan (1-iya, 0-tidak)
18. Online_only : indikasi produk hanya dijual secara online atau tidak (1-iya, 0-tidak)
19. Out_of_stock : indikasi produk sedang kehabisan stok atau tidak (1-iya, 0-tidak)
20. Sephora_exclusive : indikasi produk adalah barang ekslusif Sephora atau bukan (1-iya, 0-tidak)
21. Highlights : list berisi tag atau fitur yang menonjolkan atribut produk (contoh: vegan, matte finish)
22. Primary_category : kategori pertama dibagian breadcrumb
23. Secondly_category : kategori kedua dibagian breadcrumb
24. Tertiary_category : kategori ketiga dibagian breadcrumb
25. Child_count : jumlah variasi dari ketersediaan produk
26. Child_mac_price : harga tertinggi dari variasi produk
27. Child_min_price : harga terendah dari variasi produk


Dengan tipe data sebagai berikut:
* Terdapat 12 tipe data object
* Terdapat 8 tipe data int64
* Terdapat 7 tipe data float64

#### Cek Isi Baris
"""

data.head()

"""#### Cek Deskripsi Statistik

Dilakukan untuk mengetahui distribusi awal data, mengidentifikasi outlier, dan melihat apakah data memiliki variansi atau semuanya sama.
"""

data.describe(include='all')

"""Fungsi describe digunakan untuk memberikan informasi statistik.

* Count adalah jumlah sampel pada data.
* Mean adalah nilai rata-rata.
* Std adalah standar deviasi.
* Min yaitu nilai minimum.
* 25% adalah kuartil pertama.
* 50% adalah kuartil kedua.
* 75% adalah kuartil ketiga.
* Max adalah nilai maksimum.

#### Cek Data Hilang dan Penanganannya

Melakukan cek missing value di data understanding berguna agar data yang bersih, saat data memasuki tahap EDA sudah tidak ada anomali yang dapat mengganggu hasil dari EDA nya.
"""

missing_values = data.isna().sum()
print(missing_values)

"""Terlihat bahwa dalam dataset terdapat missing value.
Berikut penanganan yang sesuai dengan tipe data masing-masing fitur:
- Untuk fitur rating dan reviews menggunakan imputasi dengan median. Kkarena fitur ini termasuk fitur numerikal, maka imputasi yang dilakukan menggunakan median.
- Untuk fitur size, variation_type, variation_value, ingredients, highlights, secondary_category, tertiary_category imputasi dengan string "Unknown". Karena fitur tersebut termasuk fitur kategorikal, maka imputasi yang dilakukan menggunakan string.
- Untuk variation_desc, value_price_usd, sale_price_usd, child_min_price, child_max_price akan dihapus karena terlalu banyak missing value.

Walaupun penanganan missing value termasuk tahap data preparation, namun tetap dilakukan sebelum EDA agar hasil dari EDA dapat memberikan hasil yang maksimal.
"""

# Menghapus fitur yang memiliki banyak missing value >85% kosong.
data = data.drop(['variation_desc', 'value_price_usd', 'sale_price_usd', 'child_max_price', 'child_min_price'], axis=1)

# Imputasi semua missing values numerik kontinu dengan median
data.fillna(data.median(numeric_only=True), inplace=True)
print(data.isnull().sum())

# Imputasi semua missing values kategorik dengan string Unknown
data.fillna({
    'ingredients': 'Unknown',
    'variation_type': 'Unknown',
    'variation_value': 'Unknown',
    'size': 'Unknown',
    'highlights': 'Unknown',
    'secondary_category': 'Unknown',
    'tertiary_category': 'Unknown'
}, inplace=True)
print(data.isnull().sum())

"""Setelah imputasi dilakukan, sudah tidak ada lagi data yang hilang.

#### Cek Duplikasi Data

Dilakukan untuk menghindari bias dari data yang muncul dua kali.
"""

duplicates = data.duplicated()
print("Baris duplikat:", duplicates.sum())

"""Tidak terdapat data yang terduplikasi

### EDA - Univariate Analysis

Dilakukan untuk memahami persebaran nilai, baik dari fitur numerik maupun fitur kategorikal.
"""

numeric_features = data.select_dtypes(include=['number']).columns
plt.figure(figsize=(12, 100))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(13, 1, i)
    sns.histplot(data[col], bins=30, kde=True)
    plt.title(f"Distribusi {col}")
plt.tight_layout()
plt.show()

categorical_features = data.select_dtypes(include=['object']).columns
for col in categorical_features:
    print(f"{col}: {data[col].nunique()} kategori unik")

"""Fitur kategori tidak semua memungkinkan untuk ditampilkan dalam bentuk diagram batang, selain karena jumlah fiturnya lebih dari 50, tapi juga karena isi dari fitur bisa berupa teks panjang."""

selected_categorical_features = ['variation_type', 'primary_category', 'secondary_category']
plt.figure(figsize=(16, 20))
for i, col in enumerate(selected_categorical_features, 1):
    plt.subplot(3, 1, i)
    sns.countplot(x=data[col])
    plt.title(f"Distribusi {col}")
    plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""Pada distribusi fitur kategorikal hanya menampilkan kolom variation_type, primary_category, dan secondary_category, karena fitur lainnya memiliki banyak jenis yang membuatnya tidak memungkinkan untuk ditampilkan dalam bentuk subplot.

Pada variation_type, size menjadi data yang paling sering muncul.
Pada primary_category, makeup dan skincare menjadi data yang paling sering muncul.
Pada secondary_category, women menjadi data yang paling sering muncul.

### Correlation Matrix

Digunakan untuk melihat fitur apa saja yang memiliki korelasi.
"""

correlation = data[numeric_features].corr()
plt.figure(figsize=(18, 12))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Heatmap Korelasi Antar Variabel Numerik")
plt.show()

"""Terlihat pada metrik korelasi bahwa setiap fitur numerik tidak memiliki korelasi satu sama lain, kecuali fitur reviews dan loves_count. Kedua fitur tersebut berkorelasi positif, yang artinya jika fitur reviews meningkat maka fitur loves_count juga ikut meningkat.

## Data Preparation

Teknik yang akan dilakukan:
* Fitur selection : menghapus beberapa fitur yang tidak relevan
* Fitur Combine : menggabungkan beberapa fitur menjadi satu fitur
* Text Preprocessing : membersihkan data dari simbol, tanda baca, atau kata yang tidak terlalu bermakna
* TF-IDF Vectorizer

### Fitur Selection

Dilakukan untuk memilih fitur-fitur yang paling relevan dengan sistem rekomendasi, tujuannya untuk meningkatkan akurasi dan generalisasi model.
"""

print(data.columns.tolist())

# Contoh drop kolom dengan korelasi sangat tinggi
data = data.drop(['brand_id', 'loves_count', 'rating', 'reviews', 'size', 'variation_type', 'variation_value', 'price_usd', 'limited_edition', 'new', 'online_only', 'out_of_stock', 'sephora_exclusive', 'child_count'], axis=1)

"""Kenapa banyak fitur yang dihapus?
Karena sistem rekomendasi ini sangat sederhana dan berbasis teks. Fitur yang dipilih pun akan cocok dengan pendekatan Content-Based Filtering dengan menggunakan TF-IDF dan Cosine Similarity.
"""

data.info()

"""Alasan mengapa hanya fitur tersebut yang dipilih:
- **product_name** : akan digunakan sebagai identitas utama produk dan input dari pengguna untuk mencari rekomendasi
- **brand_name** : untuk memberikan informasi penting tentang produk dan pengguna dapat memilih brand sesuai dengan preferensi mereka
- **ingredients** : isi dari fitur ini penting untuk membandingkan produk
- **highlights** : berguna untuk mencocokan kebutuhan pengguna, karena berisi rangkuman poin penting dan manfaat produk
- **primary_category, secondary_category, tertiary_category** : digunakan untuk memahami jenis produk, dari umum ke spesifik
- **product_id** : fitur ini tidak masuk ke dalam pemodelan tapi akan digunakan sebagai referensi unik untuk identifikasi produk saat menampilkan hasil

### Fitur Combine

Dilakukan untuk membuat fitur baru dari kombinasi fitur lama untuk memperkuat sinyal dalam data. Tujuannya untuk menyediakan informasi tambahan yang tidak langsung terlihat dari fitur asli dan menangkap interaksi antar fitur, Ini akan memudahkan model dalam mengidentifikasi dua produk yang serupa, tapi beda brand.
"""

data['category'] = (data['primary_category'] + ' ' +
                    data['secondary_category'] + ' ' +
                    data['tertiary_category'])

"""Menggabungkan fitur primary_category, secondary_category, dan tertiary_category menjadi satu fitur, yaitu category."""

data['combined_text'] = (
    data['brand_name'] + ' ' +
    data['category'] + ' ' +
    data['highlights'] + ' ' +
    data['ingredients'])

"""Menggabungkan fitur brand_name, category, highlights, dan ingredients menjadi satu fitur, yaitu combined_text.

### Text Preprocessing

Tahap ini dilakukan agar data yang ada bersih dari karakter, simbol, atau kata yang tidak bermakna dan saat diaplikasikan ke model hasilnya dapat maksimal.
"""

stop_words = set(stopwords.words('english'))

# Fungsi preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Terapkan ke kolom
data['cleaned_text'] = data['combined_text'].apply(clean_text)

"""```stop_words = set(stopwords.words('english')) ```
Kode tersebut mengambil daftar kata umu dalam bahasa inggris dan disimpan sebagai set untuk pencarian yang lebih tepat, yang digunakan untuk menghapus kata-kata yang tidak penting dari teks.

Fungsi ``` def clean_text(text) ``` : memiliki satu parameter, yaitu text dan berisi beberapa perintah, diantaranya adalah:
- ``` text.lower() ``` : digunakan untuk memngubah seluruh huruf dalam teks menjadi huruf kecil
- ``` re.sub(r'[^a-z\s]', '', text) ``` : digunakan untuk menghapus semua karakter selain huruf a-z dan spasi
- ``` text.split() ``` : digunakan untuk memecah teks menjadi daftar kata-kata berdasarkan spasi
- ``` [word for word in tokens if word not in stop_words] ``` : digunakan untuk menghapus stopwords dari daftar text
- ``` return ' '.join(tokens) ``` : digunakan untuk menggabungkan token yang tersisa menjadi satu kalimat string

Jadi, fungsi ```clean_text()``` adalah untuk menyamakan bentuk huruf, menghapus simbol atau angka, menghapus kata yang tidak bermakna, memecah kata, dan mengembalikannya menjadi teks bersih.

### TF-IDF Vektorizer

TF-IDF adalah metode konversi teks menjadi angka yang mencerminkan pentingnya suatu kata dalam sebuah dokumen relatif terhadap seluruh korpus

Kelebihan:
- Sederhana dan cepat digunakan
- Mengurangi bobot kata umum
- Fokus pada kata yang relevan

Kekurangan:
- Tidak memahami konteks
- Tidak menangkap urutan kata
- Tidak memperhitungkan sinonim
"""

# Buat TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Ubah teks ke dalam bentuk matriks TF-IDF
tfidf_matrix = tfidf.fit_transform(data['cleaned_text'])

"""Penjelasan kode di atas:
- Variabel ``` tfidf ``` bertugas untuk mengubah teks menjadi angka dan menghapus stopwords. TF-IDF memberi bobot lebih besar pada kata yang sering muncul dalam satu dokumen tapi jarang muncul di semua dokumen, sehingga lebih informatif.
- Variabel ``` tfidf_matrix ``` bertugas untuk menerapkan TF-IDF pada kolom ``` cleaned_text ```, sehingga outputnya adalah angka-angka TF-IDF untuk digunakan dalam perhitungan kemiripan.

Jadi, inti dari kedua kode di atas adalah mengubah teks ke dalam bentuk angka berbobot yang bisa diproses oleh komputer untuk mencari kemiripan antar produk.

## Data Modelling

Hal yang dilakukan:
- Cosine Similarity
- Pengujian Model

### Cosine Similarity

Cosine similarity adalah ukuran kesamaan antara dua vektor berdasarkan sudut kosinus di antara mereka, bukan nilai absolut atau jarak euclidean. Fungsi utamanya untuk mengukur kemiripan dokumen teks, terutama yang direpresentasikan dalam bentuk TF-IDF.

Kelebihan:
- Tidak dipengaruhi oleh panjang dokumen
- Cepat dihitung dan efisien

Kekurangan:
- Tidak mempertimbangkan frekuensi kata yang tinggi
- Tidak memahami makna kata
"""

cosine_sim = cosine_similarity(tfidf_matrix)
cosine_sim

"""``` cosine_similarity ``` digunakan untuk menghitung kemiripan kosinus antara semua pasangan produk berdasarkan representasi TF-IDF. Jadi, kode ini menghitung seberapa mirip setiap produk dengan produk lain berdasarkan fitur teksnya. Tujuannya untuk menemukan produk yang paling mirip dengan produk tertentu."""

cosine_sim_df = pd.DataFrame(cosine_sim, index=data['product_name'], columns=data['product_name'])
print('Shape:', cosine_sim_df.shape)

cosine_sim_df.sample(5, axis=1).sample(10, axis=0)

"""### Pengujian Model"""

pd.set_option('display.max_colwidth', 200)
def shorten_text(text, max_len=200):
    return text if len(text) <= max_len else text[:max_len-3] + "..."

def recommend(product_name, category=None, cosine_sim=cosine_sim):
    matches = data[data['product_name'].str.contains(product_name, case=False, na=False)]

    if matches.empty:
        return "Produk tidak ditemukan. Coba masukkan sebagian kata dari nama produk."

    print("Produk yang cocok dengan pencarianmu:")
    print(matches['product_name'].head(3).to_string(index=False))

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]

    product_indices = [i[0] for i in sim_scores]
    recommendations = data[['product_name', 'brand_name', 'category', 'ingredients', 'highlights']].iloc[product_indices]

    # Filter kategori jika diberikan
    if category:
        recommendations = recommendations[recommendations['category'].str.contains(category, case=False, na=False)]

    if recommendations.empty:
        return "Tidak ada produk rekomendasi yang cocok dengan filter kategori tersebut."

    recommendations['product_name'] = recommendations['product_name'].apply(shorten_text)
    recommendations['brand_name'] = recommendations['brand_name'].apply(shorten_text)
    recommendations['category'] = recommendations['category'].apply(shorten_text)
    recommendations['ingredients'] = recommendations['ingredients'].apply(shorten_text)
    recommendations['highlights'] = recommendations['highlights'].apply(shorten_text)
    print("\nRekomendasi produk untukmu:")
    return recommendations

"""Penjelasan kode di atas:
- Fungs ``` shorten_text ``` : memiliki dua parameter, yaitu:
  - text : berisi teks yang akan dipersingkat
  - max_len : panjang maksimum karakter teks yang diizinkan.
  
  Fungsi tersebut berfungsi untuk memotong teks panjang agar tidak melebihi max_len karakter.
- Fungsi ``` recommend ``` memiliki tiga parameter, yaitu:
  - product_name : nama dari produk yang akan dicari rekomendasinya.
  - category : opsional, diisi kategori produk yang diinginkan untuk filter hasil rekomendasi.
  - cosine_sim : matriks kemiripan produk.
  
  Fungsi tersebut memiliki beberapa perintah sebagai berikut:
  - variabel ``` matches ``` : berfungsi untuk mencari baris dalam data di mana kolom product_name mengandung string product_name.
  - ``` if matches.empty ``` : kondisi jika tidak ditemukan produk yang cocok, maka menghentikan fungsi dan mengembalikan pesan "Produk tidak ditemukan. Coba masukkan sebagian kata dari nama produk.".
  - ``` print("Produk yang cocok dengan pencarianmu:") ```
    
    ``` print(matches['product_name'].head(3).to_string(index=False)) ``` : untuk menampilkan maksimal 3 nama produk yang cocok.
  - ``` idx = matches.index[0] ``` : mengambil index baris pertama dari hasil pencarian matches, dimana index ini akan digunakan untuk mengambil skor cosine similarity.
  - ``` sim_scores = list(enumerate(cosine_sim[idx])) ``` : mengambil baris ke-idx dari cosine_sim yang berisi daftar kemiripan produk dengan semua produk.
  - ``` sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) ``` : mengurutkan pasangan (index, score) berdasarkan skor kemiripan dari yang paling tinggi ke paling rendah.
  - ``` sim_scores = sim_scores[1:11] ``` : mengambil 10 skor teratas, kecuali skore pertama.
  - ``` product_indices = [i[0] for i in sim_scores] ``` : mengambil hanya index produk dari sim_score.
  - ``` recommendations = data[['product_name', 'brand_name', 'category']].iloc[product_indices] ``` : mengambil kolom product_name, brand_name, dan category dari baris-baris yang indexnya paling mirip.
  - ``` if category: ``` : kondisi jika user memberikan filter category.
  - ``` if recommendations.empty: ``` : kondisi jika hasil filter kategori kosong.
  - ``` recommendations['product_name'] = recommendations['product_name'].apply(shorten_text) ```

    ```recommendations['brand_name'] = recommendations['brand_name'].apply(shorten_text) ```

    ``` recommendations['category'] = recommendations['category'].apply(shorten_text) ```
  : menerapkan fungsi shorten_text() ke masing-masing kolom agar teks panjang dipersingkat.
  - ``` return recommendations ``` : mengembalikan tabel hasil rekomendasi dalam bentuk dataframe.

Contoh penggunaan:
"""

recommend('Cleanser', category='skincare')

"""Sistem memberikan 10 rekomendasi produk yang mirip dengan Vitamin C Serum."""

recommend('Mascara', category='makeup')

"""Dapat dilihat pada pengujian di atas, 10 rekomendasi produk teratas yang ditampilkan memiliki kemiripan dengan parameter yang digunakan.

## Evaluasi

### Precision

Untuk Content-Based Filtering digunakan metrik evaluasi precision untuk melihat seberapa baik sistem rekomendasi menggunakan Cosine Similarity. Precision adalah metrik evaluasi yang akan mengukur seberapa banyak dari hasil yang ditampilkan oleh sistem benar-benar relevan. Precision memberikan gambaran seberapa banyak dari prediksi yang dikategorikan positif benar-benar merupakan kasus positif yang sebenarnya.

![Screenshot 2025-05-29 010903.png](data:image/png;base64,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)


Dimana:
- TP (True Positive), jumlah kejadian positif yang diprediksi dengan benar.
- FP (False Positive), jumlah kejadian positif yang diprediksi dengan salah.

Karena dalam pengujian dilakukan dua kali, maka untuk menghitung precision akan menggunakan average precision. Average precision dapat memberikan informasi terkait rata-rata performa sistem rekomendasi.

![image.png](data:image/png;base64,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)

Dimana:
- n : jumlah pengujian
- Precision@10𝑖: nilai precision dari setiap pengujian.


Pada pengujian pertama:
recommend('Cleanser', category='skincare') menghasilkan 7 rekomendasi, dimana ada 2 rekomendasi yang tidak relevan. Artinya, pengujian pertama memiliki precision sebesar **0.7**.

Pada pengujian kedua:
recommend('Waterproof', category='makeup') menghasilkan 10 rekomendasi, dimana ada 1 rekomendasi yang tidak relevan. Artinya, pengujian pertama memiliki precision sebesar **0.9**.

Rata-rata precision:
Pada pengujian pertama didapat hasil 0.7 dan pada pengujian kedua didapat hasil 0.9, maka rata-rata precision adalah sebesar **0.8**.

## Dampak Model terhadap Business Understanding
### Apakah Model Menjawab Problem Statements?
* **Ya**, model merekomendasikan produk yang relevan dengan bantuan dari TF-IDf dan Cosine Similarity, dimana pengguna hanya perlu memasukan yang ingin direkomendasikan dan memasukan kategori yang relevan dengan preferensi pengguna.

### Apakah Model Berhasil Mencapai Goals?
* **Ya**, model berhasil menciptakan sistem rekomendasi dan memberikan Top 10 Recommendation berdasarkan dengan produk dan kategori produk.

### Apakah Solusi yang Direncanakan Berdampak?
* **Ya**, solusi berdampak pada model dan hasil dari model. dengan penggabungan fitur, sistem dapat mengeneralisir
"""