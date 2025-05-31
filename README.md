# Laporan Proyek Machine Learning - Salsa Tashfiyatul Qolbi

## Project Overview
Pada era ini, pengguna dihadapkan pada ribuan pilihan produk dengan berbagai manfaat, kandungan, dan klaim. Industri kecantikan dan perawatan kulit berlomba-lomba membuat produk terbaru dan memiliki banyak manfaat serta klaim. Hal ini dapat menimbulkan kebingungan dalam memilih produk yang paling sesuai dengan kebutuhan dan preferensi masing-masing individu. Oleh karena itu, dibutuhkan sebuah sistem yang mampu membantu pengguna menemukan produk yang relevan secara efisien dan personal. Salah satu solusi yang dapat diterapkan adalah sistem rekomendasi berbasis konten (content-based recommendation system), yang menyarankan produk berdasarkan kemiripan informasi produk lain yang telah diketahui.

Proyek ini bertujuan untuk membangun sistem rekomendasi produk makeup dan skincare menggunakan pendekatan content-based filtering. Sistem ini mengekstrak fitur penting dari produk seperti deskripsi, bahan (ingredients), kategori, dan jenis produk, lalu menghitung kemiripan antar produk menggunakan teknik TF-IDF vectorization dan cosine similarity. Dengan pendekatan ini, sistem dapat memberikan rekomendasi top-10 produk serupa kepada pengguna, tanpa riwayat interaksi sebelumnya.

## Business Understanding

### Problem Statements
* Bagaimana cara merekomendasikan produk skincare yang mirip berdasarkan produk yang sedang dilihat pengguna di platform e-commerce Sephora?
* Bagaimana sistem rekomendasi dapat membantu pengguna menemukan produk yang relevan meskipun tanpa riwayat interaksi pengguna sebelumnya?

### Goals
* Membangun sistem rekomendasi berbasis konten (Content-Based Filtering) yang memberikan top-N rekomendasi produk skincare yang mirip, berdasarkan konten produk seperti deskripsi, bahan, dan kategori.
* Menyediakan antarmuka pengguna sederhana yang dapat menampilkan hasil rekomendasi dengan jelas dan dapat diinterpretasikan.

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

![num_distribution](https://github.com/salstq/Recommendation-System-Products-and-Skincare-/blob/main/Gambar/Numerikal_distribution.png)

Berikut adalah persebaran data atau distribusi data dari fitur kategorikal.

![cate_distribution](https://github.com/salstq/Recommendation-System-Products-and-Skincare-/blob/main/Gambar/Categorikal_distribution.png)

### Correlation Matrix

![corr](https://github.com/salstq/Recommendation-System-Products-and-Skincare-/blob/main/Gambar/Corelation.png)

Terlihat pada metrik korelasi bahwa setiap fitur numerik tidak memiliki korelasi satu sama lain, kecuali fitur reviews dan loves_count. Kedua fitur tersebut berkorelasi positif, yang artinya jika fitur reviews meningkat maka fitur loves_count juga ikut meningkat.

## Data Preparation
Teknik yang akan dilakukan:
- Cek Missing Value & Duplikasi
- Fitur selection : menghapus beberapa fitur yang tidak relevan
- Fitur Combine : menggabungkan beberapa fitur menjadi satu fitur
- Text Preprocessing : membersihkan data dari simbol, tanda baca, atau kata yang tidak terlalu bermakna
- TF-IDF Vectorization

### Check Missing Value dan Duplikasi

| Column   | Duplicate Count | Missing Values Count |
|----------|-----------------|----------------------|
| product_id        | 0 |       0 | 
| product_name      | 0 |       0 |
| brand_id          | 0 |       0 |
| brand_name        | 0 |       0 |
| loves_count       | 0 |       0 |
| rating            | 0 |     278 |
| reviews           | 0 |     278 |
| size              | 0 |    1631 |
| variation_type    | 0 |    1444 |
| variation_value   | 0 |    1598 |
| variation_desc    | 0 |    7244 |
| ingredients       | 0 |     945 |
| price_usd         | 0 |       0 |
| value_price_usd   | 0 |    8043 |
| sale_price_usd    | 0 |    8224 |
| limited_edition   | 0 |       0 |
| new               | 0 |       0 |
| online_only       | 0 |       0 |
| out_of_stock      | 0 |       0 |
| sephora_exclusive | 0 |       0 |
| highlights        | 0 |    2207 |
| primary_category  | 0 |       0 |
| secondary_category| 0 |       8 |
| tertiary_category | 0 |     990 |
| child_count       | 0 |       0 |
| child_max_price   | 0 |    5740 |
| child_min_price   | 0 |    5740 |

Walaupun penanganan missing value termasuk tahap data preparation, namun tetap dilakukan sebelum EDA agar hasil dari EDA dapat memberikan hasil yang maksimal. Terlihat bahwa dalam dataset terdapat missing value. Berikut penanganan yang sesuai dengan tipe data masing-masing fitur:
- Untuk fitur rating dan reviews menggunakan imputasi dengan median. Karena fitur ini termasuk fitur numerikal, maka imputasi yang dilakukan menggunakan median.
- Untuk fitur size, variation_type, variation_value, ingredients, highlights, secondary_category, tertiary_category imputasi dengan string "Unknown". Karena fitur tersebut termasuk fitur kategorikal, maka imputasi yang dilakukan menggunakan string.
- Untuk variation_desc, value_price_usd, sale_price_usd, child_min_price, child_max_price akan dihapus karena terlalu banyak missing value.

Tidak terdapat data yang terduplikasi. Sama hal nya dengan missing value, cek duplikasi data dilakukan sebelum EDA.

### Fitur Selection
Dilakukan untuk memilih fitur-fitur yang paling relevan dengan sistem rekomendasi, tujuannya untuk meningkatkan akurasi dan generalisasi model. Kenapa banyak fitur yang dihapus? Karena sistem rekomendasi ini sangat sederhana dan berbasis teks. Fitur yang dipilih pun akan cocok dengan pendekatan Content-Based Filtering dengan menggunakan TF-IDF dan Cosine Similarity.

### Fitur Combine
Dilakukan untuk membuat fitur baru dari kombinasi fitur lama untuk memperkuat sinyal dalam data. Tujuannya untuk menyediakan informasi tambahan yang tidak langsung terlihat dari fitur asli dan menangkap interaksi antar fitur, Ini akan memudahkan model dalam mengidentifikasi dua produk yang serupa, tapi beda brand.

### Text Preprocessing
Tujuan dari tahap ini adalah untuk menyamakan bentuk huruf, menghapus simbol atau angka, menghapus kata yang tidak bermakna, memecah kata, dan mengembalikannya menjadi teks bersih.

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
  
cosine_similarity() digunakan untuk menghitung kesamaan antar item berdasarkan representasi TF-IDF (Term Frequency-Inverse Document Frequency) yang telah dihitung sebelumnya. Setelah digunakan fungsi tersebut, maka dihasilkan dataframe seperti contoh sebagai berikut.

| product_name |	Vital Perfection Intensive WrinkleSpot Treatment |	Glass & Gloss 2-Step Facial Retexturizing & Brightening Treatment |	Supa Thick Scalp Serum |	Mini Glowy Super Gel Lightweight Dewy Highlighter |	Brow Flick Microfine Detailing Eyebrow Pen |
|------------------|-----------------|----------------------|-----------------|-------------------|------------------|
|		The Clarifying Kit |	0.107545 |	0.197122 |	0.135246 |	0.113965 |	0.128624 |
| Sparkling Cuvee Glass Home Diffuser |	0.025888 |	0.000000 |	0.019217 |	0.012491 |	0.035391 |
| NuFACE Mini+ Petite Facial Toning Device |	0.186921 |	0.187842 |	0.182397 |	0.066684 |	0.128531 |
| Strength Cure Blonde Purple Shampoo |	0.155876 |	0.163121 |	0.180727 |	0.112218 |	0.122465 |
| Book Expressive |	0.035719 |	0.059015 |	0.050872 |	0.039107 |	0.049795 |
| Pink Cloud Soft Moisture Cream |	0.176780	|0.090978	|0.212678	|0.098747	|0.051685 |
| Mini Pore Minimizing Instant Detox Mask	| 0.119873	|0.085177	|0.149354	|0.106068	|0.123096 |
| LUNA play plus 2 |	0.038280	|0.073430	|0.056964	|0.000000	|0.000000 |
| Glam Guard Long-Wear Setting Spray|	0.069566 |	0.085817	|0.047980	|0.129215	|0.160556 |
| Bloom Eau de Parfum Intense |	0.060322 | 0.035526	|0.034713	|0.015809	|0.018780 |

**Penjelasan**: Setiap kolom berikutnya menggambarkan tingkat kemiripan antara produk yang memiliki kemiripan dengan produk lainnya berdasarkan faktor-faktornya dimana nilai pada setiap cell menunjukkan tingkat kemiripan, yang dihitung berdasarkan perbandingan faktor-faktor antara dua film. Nilai 0 menunjukkan bahwa tidak ada kemiripan antara produk tersebut.

### Pengujian Model

Dilakukan 2x pengujian sebagai berikut:
1. `recommend('Cleanser', category='skincare')`

Produk yang cocok dengan pencarianmu:<br>
- GENIUS Ultimate Anti-Aging Melting Cleanser
- Gentle Rejuvenating Cleanser
- Balancing Cleanser

Rekomendasi produk untukmu:
|	| product_name |	brand_name	 |	category	 |	ingredients	 |	highlights |	
|	-------------- |	----------------- |	------------------ |	------------------ |	------------------- | ------------- |		
|	1096 |		Vinoclean Makeup Removing Cleansing Oil  |		Caudalie |		Skincare Cleansers Face Wash & Cleansers |	['Helianthus Annuus (Sunflower) Seed Oil*, Polyglyceryl-4 Oleate*, Caprylic/Capric Triglyceride*, Ricinus Communis (Castor) Seed Oil*, Prunus Amygdalus Dulcis (Sweet Almond) Oil*, Vitis Vinifera (...	 |	['Vegan', 'Clean + Planet Positive', 'Without Mineral Oil', 'Good for: Dryness'] |	
 |	3555 |		Green Tea Hydrating Cleansing Oil |		innisfree	 |	Skincare Cleansers Face Wash & Cleansers	 |	 ['Triethylhexanoin, Isopropyl Palmitate, Caprylic/Capric Triglyceride, Sorbeth-30 Tetraoleate, Cetyl Ethylhexanoate, Olea Europaea (Olive) Fruit Oil, Cocos Nucifera (Coconut) Oil, Fragrance / Parf...	 |	['Hydrating', 'Without Mineral Oil', 'Good for: Dryness', 'Without Parabens', 'Best for Dry, Combo, Normal Skin', 'Without Sulfates SLS & SLES'] |	
 |	2238	 |	Slaai Makeup-Melting Butter Cleanser	 |	Drunk Elephant	 |	Skincare Cleansers Face Wash & Cleansers |		['Ethylhexyl Palmitate, Caprylic/Capric Triglyceride, Carthamus Tinctorius (Safflower) Seed Oil, Lauryl Laurate, Polyglyceryl-3 Laurate, Polyhydroxystearic Acid, Helianthus Annuus (Sunflower) Seed... |		Unknown |	
 |	1281 |		Multi-Miracle Glow Cleansing Balm	 |	Charlotte Tilbury	 |	Skincare Cleansers Face Wash & Cleansers |		['Glycerin, Water, Caprylic/Capric Triglyceride, Cyclopentasiloxane, Sucrose Stearate, Phenyl Trimethicone, Phenoxyethanol, Microcrystalline Cellulose, Ethylhexylglycerin, Chlorphenesin, Cellulose... |		Unknown |	
 |	98 |		Advanced Anti-Aging Repairing Oil	 |	Algenist |		Skincare Moisturizers Face Oils	 |	['Chlorella Protothecoides Oil, Cetearyl Ethylhexanoate, Isopropyl Isostearate, Caprylic/Capric Triglyceride, Ceramide 3, Alaria Esculenta Extract, Retinyl Palmitate, Tocopherol, Rosmarinus Offici...	 |	Unknown |	
 |	558 |		The POREfessional Get Unblocked Makeup-Removing Cleansing Oil	 |	Benefit Cosmetics	 |	Skincare Cleansers Face Wash & Cleansers |		['Caprylic/Capric Triglyceride, Simmondsia Chinensis (Jojoba) Seed Oil, PEG-40 Sorbitan Peroleate, Persea Gratissima (Avocado) Oil, Vitis Vinifera (Grape) Seed Oil, 1,2-Hexanediol, Caprylyl Glycol... |		['Good for: Dullness/Uneven Texture', 'Good for: Pores'] |	
 |	4159  |		Ultra Facial Cream Refill Bundle	 |	Kiehl's Since 1851	 |	Skincare Value & Gift Sets Unknown	 |	['Aqua / Water Glycerin Dimethicone Squalane Bis-Peg-18 Methyl Ether Dimethyl Silane Sucrose Stearate Stearyl Alcohol Peg-8 Stearate Myristyl Myristate Prunus Armeniaca Kernel Oil / Apricot Kernel...	 |	['Clean at Sephora', 'Good for: Dryness', 'Without Parabens'] |	

2. `recommend('Mascara', category='makeup')`
  
Produk yang cocok dengan pencarianmu:<br>
- Lash-Amplifying Volumizing & Lengthening Mascara
- Lash Brag Jet-Black Volumizing Mascara
- Mini Lash Brag Jet-Black Volumizing Mascara

Rekomendasi produk untukmu:
| |  product_name | 	brand_name	| category| 	ingredients| 	highlights| 
|----------------|------------------|---------------------|-----------------|-----------------|---------------|
| 4951	 | Brow 1980 Volumizing Eyebrow Pomade Gel | 	MERIT	 | Makeup Eye Eyebrow	 | ['Aqua (Water, Eau), Mica, Cera Carnauba (Copernicia Cerifera (Carnauba) Wax, Cire de Carnauba), Polyisobutene, Propanediol, Glyceryl Stearate, Polyacrylate-21, Kaolin, Palmitic Acid, Oleic Acid, ...	 | ['Cruelty-Free', 'Volumizing', 'Vegan', 'Long-wearing', 'Fragrance Free', 'Clean at Sephora'] | 
| 3349	 | LEGIT LASHES Double-Ended Volumizing and Lengthening Mascara	 | HUDA BEAUTY	 | Makeup Eye Mascara | 	['Volume:', 'Water/Aqua/Eau, Beeswax/Cera Alba/Cire d’abeille, Stearic Acid, Helianthus Annuus (Sunflower) Seed Wax, Glyceryl Behenate, Acacia Senegal Gum, Polyglyceryl-6 Distearate, Vp/Eicosene C...	 | ['Matte Finish', 'Volumizing', 'Curling', 'Lengthening'] | 
| 4770	| The Professionall 24HR Double-Ended Lifting & Volumizing Mascara | 	MAKE UP FOR EVER | 	Makeup Eye Mascara | 	['Step 1:', 'Aqua (Water), Candelilla Cera (Euphorbia Cerifera (Candelilla) Wax), Synthetic Beeswax, Vp/Eicosene Copolymer, Paraffin, Polyacrylate-21, Copernicia Cerifera (Carnauba) Wax, Stearyl A...	 | ['Volumizing', 'Long-wearing', 'Lengthening', 'Curling'] | 
| 3369 |	Mini LEGIT LASHES Double-Ended Volumizing and Lengthening Mascara | 	HUDA BEAUTY	 | Mini Size Makeup Unknown	 | ['Volume:', 'Water/Aqua/Eau, Beeswax/Cera Alba/Cire d’abeille, Stearic Acid, Helianthus Annuus (Sunflower) Seed Wax, Glyceryl Behenate, Acacia Senegal Gum, Polyglyceryl-6 Distearate, Vp/Eicosene C...	| ['Matte Finish', 'Volumizing', 'Curling', 'Lengthening'] | 
| 3037	 | L'Obscur Lengthening Mascara	 | Gucci	 | Makeup Eye Mascara	 | ['Aqua/Water/Eau, Synthetic Beeswax, Paraffin, Stearic Acid, Acacia Senegal Gum, Triethanolamine, Butylene Glycol, Copernicia Cerifera Cera/Copernicia Cerifera (Carnauba) Wax/Cire de Carnauba, Pol...	 | ['allure 2020 Best of Beauty Award Winner', 'Long-wearing', 'Lengthening', 'Without Parabens'] | 
| 6256	 | Perfect Strokes Universal Volumizing Mascara	 | Rare Beauty by Selena Gomez	 | Makeup Eye Mascara	|['Water/Aqua/Eau, Copernicia Cerifera (Carnauba) Wax/Cera Carnauba/Cire de carnauba, Glyceryl Stearate, Hydrogenated Olive Oil Stearyl Esters, Synthetic Wax, Stearic Acid, Helianthus Annuus (Sunfl...	 | ['Volumizing', 'Curling', 'Lengthening', 'Long-wearing', 'Without Parabens', 'Vegan'] | 
| 3117 | 	Noir G Volumizing & Curling Mascara	 | GUERLAIN	 | Makeup Eye Mascara	 | ['Aqua (Water), Olea Europaea (Olive) Oil Unsaponifiables, Cera Alba (Beeswax), Polyacrylate-21, Bis-Diglyceryl Polyacyladipate-2, Candelilla Cera (Euphorbia Cerifera (Candelilla) Wax), Cera Carna... | 	['Volumizing', 'Waterproof', 'Curling'] | 
| 3110 | 	Mad Eyes Mascara Long-Wearing & Volumizing	 | GUERLAIN	 | Makeup Eye Mascara | 	['Aqua (Water), Cera Alba (Beeswax), Propanediol, Glyceryl Stearate, Polyacrylate-21, Bis-Diglyceryl Polyacyladipate-2, Alcohol, Candelilla Cera (Euphorbia Cerifera (Candelilla) Wax), Cera Carnaub...	 | Unknown | 
| 1258	 | Legendary Lashes Volume 2 Mascara | 	Charlotte Tilbury	 | Makeup Eye Mascara	 | ['Water, Synthetic Beeswax, Paraffin, Stearic Acid, Acacia Senegal Gum, Triethanolamine, Butylene Glycol, Copernicia Cerifera (Carnauba) Wax/Cire De Carnauba, Polybutene, Vp/Eicosene Copolymer, Gl... | 	Unknown | 
| 7843 | 	Better Than Sex Volumizing & Lengthening Mascara	 | Too Faced	 | Makeup Eye Mascara	 | ['Chocolate: Water\\Aqua\\Eau, Paraffin, Glyceryl Stearate, Synthetic Beeswax, Acacia Senegal Gum, Butylene Glycol, Stearic Acid, Palmitic Acid, Polybutene, Oryza Sativa (Rice) Bran Wax, Ozokerite... | 	['Volumizing', 'Lengthening', 'Cruelty-Free'] | 

Selain menampilkan rekomendasi, sistem juga memberikan produk yang cocok atau serupa dengan produk yang dicari oleh pengguna.

## Evaluasi

### Precision
Untuk Content-Based Filtering digunakan metrik evaluasi precision untuk melihat seberapa baik sistem rekomendasi menggunakan Cosine Similarity. Precision adalah metrik evaluasi yang akan mengukur seberapa banyak dari hasil yang ditampilkan oleh sistem benar-benar relevan. Precision memberikan gambaran seberapa banyak dari prediksi yang dikategorikan positif benar-benar merupakan kasus positif yang sebenarnya.

![Screenshot 2025-05-29 010903.png](https://github.com/salstq/Recommendation-System-Products-and-Skincare-/blob/main/Gambar/Precision.png)

Dimana:
- TP (True Positive), jumlah kejadian positif yang diprediksi dengan benar.
- FP (False Positive), jumlah kejadian positif yang diprediksi dengan salah.

Karena dalam pengujian dilakukan dua kali, maka untuk menghitung precision akan menggunakan average precision. Average precision dapat memberikan informasi terkait rata-rata performa sistem rekomendasi.

![image.png](https://github.com/salstq/Recommendation-System-Products-and-Skincare-/blob/main/Gambar/Average_Precision.png)

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
* **Ya**, model menggunakan pendekatan content-based filtering yang memanfaatkan TF-IDF vectorization dan cosine similarity untuk merekomendasikan produk serupa berdasarkan deskripsi, bahan, dan kategori produk. Ini sesuai dengan problem statements yang ingin merekomendasikan produk tanpa bergantung pada riwayat interaksi pengguna.

### Apakah Model Berhasil Mencapai Goals?
* **Ya**, model berhasil memberikan rekomendasi Top 10 produk yang relevan dan serupa berdasarkan konten produk. Proses preprocessing, ekstraksi fitur, dan perhitungan kemiripan dilakukan dengan benar, dan hasil rekomendasi ditampilkan secara jelas, sesuai dengan tujuan yang ditetapkan.

### Apakah Solusi yang Direncanakan Berdampak?
* **Ya**, solusi yang direncanakan terbukti berdampak karena seluruh komponen yang dirancang berhasil diimplementasikan dan memberikan hasil yang sesuai tujuan. Sistem berhasil mengekstrak fitur penting dari produk seperti bahan, deskripsi, kategori, dan jenis produk, kemudian mengukur kemiripan antar produk menggunakan TF-IDF vectorization dan cosine similarity. Selain itu, karena sistem tidak memerlukan data pengguna lain, tetapi tetap mampu memberikan rekomendasi yang relevan untuk pengguna baru. Hal ini membuat sistem lebih fleksibel, scalable, dan efektif untuk diterapkan dalam lingkungan e-commerce.

## Kesimpulan
Sistem rekomendasi berbasis konten yang dibangun dalam proyek ini berhasil memberikan rekomendasi produk skincare yang relevan berdasarkan kemiripan produk. Dengan menggunakan teknik TF-IDF vectorization dan cosine similarity, sistem mampu menghitung tingkat kemiripan antar produk dan menghasilkan rekomendasi top-N produk serupa. Berdasarkan evaluasi, sistem rekomendasi produk & skincare dengan pendekatan Content-Based Filtering berhasil mencapai rata-rata precision 0.8, memberikan rekomendasi yang relevan berdasarkan kesamaan produk.

Sehingga dapat disimpulkan, bahwa pendekatan ini dapat dikatakan berhasil memenuhi tujuan dari dibuatnya projek ini dimana model dapat memprediksi produk berdasarkan deskripsi, bahan, dan kategori menggunakan content-based filtering.
