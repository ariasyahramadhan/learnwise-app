import re, io, warnings, urllib.request
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib, os, time
from multiprocessing import Pool

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Multi-processing setup for Sastrawi stemmer
stemmer = None

def init_worker():
    global stemmer
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

def preprocess_single(text: str) -> str:
    global stemmer
    if stemmer is None:
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return stemmer.stem(text)

def parallel_preprocess(texts, num_workers=16):
    print(f"Running preprocessing on {len(texts)} texts using {num_workers} parallel workers...")
    with Pool(num_workers, initializer=init_worker) as p:
        result = p.map(preprocess_single, texts, chunksize=100)
    return result

# ============================================================
# Similarity feature functions (highly optimized vector-like ops)
# ============================================================

def jaccard_similarity(s1, s2):
    w1, w2 = set(s1.split()), set(s2.split())
    if not w1 and not w2: return 0.0
    return len(w1 & w2) / len(w1 | w2)

def ngram_overlap(s1, s2, n):
    def get_ng(t):
        w = t.split()
        return set(zip(*[w[i:] for i in range(n)])) if len(w) >= n else set()
    ng1, ng2 = get_ng(s1), get_ng(s2)
    if not ng1 or not ng2: return 0.0
    return len(ng1 & ng2) / max(len(ng1), len(ng2))

def positional_ngram(s1, s2, n):
    w1, w2 = s1.split(), s2.split()
    if len(w1) < n or len(w2) < n: return 0.0
    bg1 = [tuple(w1[i:i+n]) for i in range(len(w1)-n+1)]
    bg2 = [tuple(w2[i:i+n]) for i in range(len(w2)-n+1)]
    return sum(1 for i, b in enumerate(bg1) if i < len(bg2) and b == bg2[i]) / max(len(bg1), len(bg2))

def word_order_score(s1, s2):
    w1, w2 = s1.split(), s2.split()
    common = set(w1) & set(w2)
    if not common: return 0.5
    pos1 = {w: i for i, w in enumerate(w1) if w in common}
    pos2 = {w: i for i, w in enumerate(w2) if w in common}
    words = list(common)[:30]
    con = dis = 0
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            wi, wj = words[i], words[j]
            if all(k in pos1 and k in pos2 for k in [wi, wj]):
                d1 = pos1[wi] - pos1[wj]
                d2 = pos2[wi] - pos2[wj]
                if d1 * d2 > 0: con += 1
                elif d1 * d2 < 0: dis += 1
    return con / (con + dis) if (con + dis) > 0 else 0.5

def lcs_ratio(s1, s2):
    w1, w2 = s1.split()[:50], s2.split()[:50]
    m, n = len(w1), len(w2)
    if not m or not n: return 0.0
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if w1[i-1]==w2[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n] / max(m, n)

def edit_distance_ratio(s1, s2):
    a, b = s1[:200], s2[:200]
    m, n = len(a), len(b)
    if not m and not n: return 1.0
    dp = list(range(n+1))
    for i in range(1, m+1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n+1):
            dp[j] = prev[j-1] if a[i-1]==b[j-1] else 1 + min(prev[j], dp[j-1], prev[j-1])
    return 1.0 - dp[n] / max(m, n)

def char_ngram_jaccard(s1, s2, n):
    def get_cng(t):
        return set(t[i:i+n] for i in range(len(t)-n+1)) if len(t) >= n else set()
    c1, c2 = get_cng(s1), get_cng(s2)
    if not c1 or not c2: return 0.0
    return len(c1 & c2) / len(c1 | c2)

def prefix_match_ratio(s1, s2, pl=3):
    w1, w2 = s1.split(), s2.split()
    if not w1 or not w2: return 0.0
    p1 = set(w[:pl] for w in w1 if len(w) >= pl)
    p2 = set(w[:pl] for w in w2 if len(w) >= pl)
    if not p1 or not p2: return 0.0
    return len(p1 & p2) / len(p1 | p2)

vectorizer_word = None
vectorizer_char = None

def token_sort_cosine(s1, s2):
    ss1 = ' '.join(sorted(s1.split()))
    ss2 = ' '.join(sorted(s2.split()))
    v1 = vectorizer_word.transform([ss1])
    v2 = vectorizer_word.transform([ss2])
    return float(1.0 - paired_cosine_distances(v1, v2)[0])

def cosine_word_batch(df, batch_size=5000):
    scores = []
    for s in range(0, len(df), batch_size):
        b = df.iloc[s:s+batch_size]
        v1 = vectorizer_word.transform(b['s1_clean'])
        v2 = vectorizer_word.transform(b['s2_clean'])
        scores.extend((1 - paired_cosine_distances(v1, v2)).tolist())
    return np.array(scores)

def cosine_char_batch(df, batch_size=5000):
    scores = []
    for s in range(0, len(df), batch_size):
        b = df.iloc[s:s+batch_size]
        v1 = vectorizer_char.transform(b['s1_clean'])
        v2 = vectorizer_char.transform(b['s2_clean'])
        scores.extend((1 - paired_cosine_distances(v1, v2)).tolist())
    return np.array(scores)

FEATURE_NAMES = [
    'cosine_word', 'cosine_char', 'jaccard', 'bigram', 'trigram', 'fourgram',
    'pos_bigram', 'pos_trigram', 'word_order', 'lcs',
    'edit_dist', 'char_bi_j', 'char_tri_j', 'prefix_match', 'token_sort_cosine',
    'len_diff', 'len_ratio', 'uniq_ratio', 'shared_count', 'len_s1', 'len_s2',
    'cosine_prod', 'jacc_x_order', 'mean_cosine', 'overlap_ratio', 'abs_len_norm'
]

def extract_features(df, label=''):
    start_time = time.time()
    if label: print(f"  [{label}] cosine word+char ...")
    cw = cosine_word_batch(df)
    cc = cosine_char_batch(df)
    
    s1_list = df['s1_clean'].tolist()
    s2_list = df['s2_clean'].tolist()
    
    if label: print(f"  [{label}] fitur leksikal ...")
    jacc  = np.array([jaccard_similarity(s1, s2) for s1, s2 in zip(s1_list, s2_list)])
    bi    = np.array([ngram_overlap(s1, s2, 2) for s1, s2 in zip(s1_list, s2_list)])
    tri   = np.array([ngram_overlap(s1, s2, 3) for s1, s2 in zip(s1_list, s2_list)])
    fg    = np.array([ngram_overlap(s1, s2, 4) for s1, s2 in zip(s1_list, s2_list)])
    pbi   = np.array([positional_ngram(s1, s2, 2) for s1, s2 in zip(s1_list, s2_list)])
    ptri  = np.array([positional_ngram(s1, s2, 3) for s1, s2 in zip(s1_list, s2_list)])
    wo    = np.array([word_order_score(s1, s2) for s1, s2 in zip(s1_list, s2_list)])
    lcs   = np.array([lcs_ratio(s1, s2) for s1, s2 in zip(s1_list, s2_list)])
    ed    = np.array([edit_distance_ratio(s1, s2) for s1, s2 in zip(s1_list, s2_list)])
    cbj   = np.array([char_ngram_jaccard(s1, s2, 2) for s1, s2 in zip(s1_list, s2_list)])
    ctj   = np.array([char_ngram_jaccard(s1, s2, 3) for s1, s2 in zip(s1_list, s2_list)])
    pfx   = np.array([prefix_match_ratio(s1, s2) for s1, s2 in zip(s1_list, s2_list)])
    
    # Fast token sort cosine
    tsc   = np.array([token_sort_cosine(s1, s2) for s1, s2 in zip(s1_list, s2_list)])
    
    ls1   = np.array([len(t.split()) for t in s1_list])
    ls2   = np.array([len(t.split()) for t in s2_list])
    ldiff = np.abs(ls1 - ls2).astype(float)
    lratio= np.minimum(ls1, ls2) / np.maximum(np.maximum(ls1, ls2), 1.0)
    shared= np.array([len(set(s1.split()) & set(s2.split())) for s1, s2 in zip(s1_list, s2_list)]).astype(float)
    
    uniq  = np.array([
        len(set(s1.split()).symmetric_difference(set(s2.split()))) / max(len(set(s1.split()) | set(s2.split())), 1.0)
        for s1, s2 in zip(s1_list, s2_list)
    ])
    
    # Cross-features
    cp  = cw * cc
    jxo = jacc * wo
    mc  = (cw + cc) / 2.0
    ovr = shared / np.maximum(ls1 + ls2, 1.0)
    aln = ldiff / np.maximum(ls1 + ls2, 1.0)
    
    res = np.column_stack([cw, cc, jacc, bi, tri, fg, pbi, ptri, wo, lcs,
                            ed, cbj, ctj, pfx, tsc,
                            ldiff, lratio, uniq, shared, ls1, ls2,
                            cp, jxo, mc, ovr, aln])
    print(f"  [{label}] Fitur selesai dalam {time.time() - start_time:.2f} detik")
    return res

if __name__ == '__main__':
    BASE_URL = "https://raw.githubusercontent.com/Wikidepia/indonesia_dataset/master/paraphrase/paws/data/final/"

    def load_paws(split_name):
        url = BASE_URL + split_name
        print(f"  Mengunduh {split_name} ...")
        with urllib.request.urlopen(url) as response:
            content = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(content), sep='\t')
        if 'label' in df.columns:
            df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(-1).astype(int)
        df = df[df['label'].isin([0, 1])].reset_index(drop=True)
        return df[['sentence1', 'sentence2', 'label']]

    print("Mengunduh dataset PAWS-Indonesia...")
    df_paws_train = load_paws("train.tsv")
    df_paws_dev   = load_paws("dev.tsv")
    df_paws_test  = load_paws("test.tsv")

    # Corpus akademik Indonesia
    akademik_pairs = [
        ("Fotosintesis adalah proses biokimia yang dilakukan tumbuhan untuk mengubah energi cahaya matahari menjadi glukosa.", "Fotosintesis merupakan proses biokimia yang dilakukan oleh tumbuhan untuk mengubah energi sinar matahari menjadi glukosa.", 1),
        ("Pancasila adalah dasar negara Republik Indonesia yang terdiri dari lima sila utama.", "Pancasila merupakan dasar negara Republik Indonesia yang terdiri atas lima sila.", 1),
        ("Pemanasan global disebabkan oleh peningkatan konsentrasi gas rumah kaca di atmosfer bumi.", "Pemanasan global terjadi akibat meningkatnya konsentrasi gas-gas rumah kaca pada lapisan atmosfer bumi.", 1),
        ("Demokrasi adalah sistem pemerintahan di mana kekuasaan tertinggi berada di tangan rakyat.", "Dalam sistem demokrasi, kekuasaan tertinggi dipegang oleh rakyat melalui mekanisme pemilihan umum.", 1),
        ("Inflasi adalah kenaikan harga barang dan jasa secara umum dan terus-menerus dalam jangka waktu tertentu.", "Inflasi merupakan peningkatan harga barang dan jasa secara keseluruhan yang berlangsung secara terus-menerus.", 1),
        ("Revolusi industri dimulai di Inggris pada abad ke-18 dan membawa perubahan besar dalam cara produksi.", "Revolusi industri bermula di Inggris pada abad ke-18 dan membawa perubahan signifikan dalam metode produksi barang.", 1),
        ("Hukum Newton menyatakan bahwa setiap benda yang diam akan tetap diam kecuali ada gaya yang bekerja padanya.", "Menurut hukum Newton, benda yang dalam keadaan diam akan tetap diam jika tidak ada gaya luar yang bekerja padanya.", 1),
        ("Pembelajaran berbasis proyek meningkatkan keterampilan berpikir kritis dan kolaboratif siswa secara signifikan.", "Pembelajaran dengan pendekatan berbasis proyek terbukti meningkatkan kemampuan berpikir kritis dan kolaborasi peserta didik.", 1),
        ("Ekosistem laut mengandung keanekaragaman hayati yang sangat tinggi dan berperan penting dalam siklus karbon global.", "Lautan memiliki keanekaragaman biologis yang sangat besar serta memainkan peran penting dalam siklus karbon di bumi.", 1),
        ("Teknologi kecerdasan buatan telah mengubah cara manusia bekerja dan berinteraksi satu sama lain di era digital.", "Kecerdasan buatan sebagai teknologi modern telah mengubah cara manusia bekerja dan berinteraksi dalam kehidupan sehari-hari.", 1),
        ("Sistem imun manusia melibatkan berbagai sel dan molekul yang bekerja bersama untuk melawan infeksi penyakit.", "Imunitas tubuh manusia melibatkan berbagai jenis sel dan molekul yang bekerja bersama dalam melawan penyakit.", 1),
        ("Keanekaragaman hayati Indonesia termasuk yang tertinggi di dunia berkat posisi geografisnya di khatulistiwa.", "Indonesia memiliki keanekaragaman hayati yang sangat tinggi karena letaknya yang berada di wilayah khatulistiwa.", 1),
        ("Globalisasi membawa dampak positif berupa kemudahan akses informasi dan pertumbuhan ekonomi yang pesat.", "Dampak positif globalisasi antara lain kemudahan dalam mengakses informasi dan pertumbuhan di sektor ekonomi.", 1),
        ("Hak asasi manusia adalah hak dasar yang dimiliki setiap individu tanpa memandang ras atau agama apapun.", "Hak asasi manusia merupakan hak dasar yang melekat pada setiap individu tanpa membedakan ras maupun agama.", 1),
        ("Literasi digital penting dikuasai generasi muda di era teknologi informasi yang terus berkembang pesat.", "Penguasaan literasi digital sangat penting bagi generasi muda di era perkembangan teknologi informasi saat ini.", 1),
        ("Kurikulum pendidikan Indonesia telah mengalami berbagai perubahan sejak kemerdekaan untuk menyesuaikan kebutuhan.", "Kurikulum pendidikan di Indonesia sudah mengalami banyak perubahan sejak Indonesia merdeka hingga saat ini.", 1),
        ("Otak manusia memiliki kemampuan plastisitas yang memungkinkan adaptasi terhadap lingkungan dan pengalaman baru.", "Plastisitas otak manusia memberi kemampuan untuk beradaptasi dengan kondisi lingkungan dan pengalaman yang berubah.", 1),
        ("Pembangunan berkelanjutan bertujuan memenuhi kebutuhan generasi kini tanpa mengorbankan generasi mendatang.", "Tujuan dari pembangunan berkelanjutan adalah memenuhi kebutuhan masa kini tanpa mengorbankan kemampuan generasi berikutnya.", 1),
        ("Teori evolusi Darwin menyatakan bahwa spesies berubah melalui proses seleksi alam dari waktu ke waktu.", "Menurut teori evolusi Darwin, perubahan spesies terjadi melalui proses seleksi alam yang berlangsung secara bertahap.", 1),
        ("Energi terbarukan seperti tenaga surya dan angin menjadi solusi ramah lingkungan untuk masa depan.", "Pemanfaatan energi terbarukan seperti tenaga surya dan angin merupakan solusi energi yang ramah lingkungan.", 1),
        ("Pertumbuhan ekonomi diukur melalui perubahan Produk Domestik Bruto suatu negara dalam periode tertentu.", "Produk Domestik Bruto digunakan sebagai indikator utama untuk mengukur pertumbuhan ekonomi suatu negara.", 1),
        ("Bahasa Indonesia ditetapkan sebagai bahasa resmi negara dalam Undang-Undang Dasar 1945 pasal 36.", "Berdasarkan UUD 1945 pasal 36, Bahasa Indonesia ditetapkan sebagai bahasa resmi Negara Kesatuan Republik Indonesia.", 1),
        ("Proses fotosintesis berlangsung di kloroplas tumbuhan dengan bantuan pigmen hijau yang disebut klorofil.", "Fotosintesis terjadi di dalam kloroplas sel tumbuhan, menggunakan pigmen klorofil yang berwarna hijau sebagai katalis.", 1),
        ("Urbanisasi adalah perpindahan penduduk dari desa ke kota yang meningkat pesat di negara berkembang.", "Fenomena urbanisasi berupa perpindahan penduduk dari wilayah pedesaan ke perkotaan terus meningkat di negara berkembang.", 1),
        ("Sistem ekonomi pasar bebas memungkinkan harga ditentukan oleh mekanisme penawaran dan permintaan pasar.", "Pada ekonomi pasar bebas, tingkat harga barang dan jasa ditentukan sepenuhnya oleh mekanisme penawaran dan permintaan.", 1),
        ("Pendidikan karakter bertujuan membentuk siswa yang berakhlak mulia dan bertanggung jawab terhadap bangsa.", "Tujuan pendidikan karakter adalah membentuk peserta didik yang berakhlak baik dan bertanggung jawab kepada bangsa.", 1),
        ("Kebijakan fiskal pemerintah mencakup pengelolaan anggaran pendapatan dan belanja negara secara efisien.", "Pengelolaan anggaran pendapatan dan belanja negara secara efisien merupakan bagian dari kebijakan fiskal pemerintah.", 1),
        ("Kualitas pendidikan tinggi di Indonesia terus ditingkatkan melalui akreditasi dan pengembangan kurikulum.", "Peningkatan kualitas perguruan tinggi Indonesia dilakukan melalui proses akreditasi dan pembaruan kurikulum secara berkala.", 1),
        ("Penelitian ilmiah harus menggunakan metodologi yang valid, reliabel, dan dapat direplikasi oleh peneliti lain.", "Metodologi penelitian yang valid dan reliabel serta dapat direplikasi oleh peneliti lain adalah syarat penelitian ilmiah.", 1),
        ("Teknologi blockchain menjamin keamanan transaksi digital melalui sistem terdesentralisasi dan transparan.", "Sistem blockchain yang terdesentralisasi dan transparan menjadi dasar keamanan transaksi digital modern.", 1),
        ("Perkembangan media sosial mengubah pola komunikasi masyarakat secara mendasar di era informasi ini.", "Era informasi saat ini ditandai oleh perubahan mendasar pola komunikasi masyarakat akibat perkembangan media sosial.", 1),
        ("Vaksin dikembangkan dalam waktu singkat berkat kemajuan teknologi bioteknologi yang revolusioner.", "Kemajuan teknologi bioteknologi yang revolusioner memungkinkan pengembangan vaksin dalam waktu yang sangat singkat.", 1),
        ("Ketahanan pangan nasional bergantung pada produktivitas pertanian, distribusi yang merata, and kebijakan impor.", "Produktivitas pertanian, distribusi pangan yang merata, dan kebijakan impor menentukan ketahanan pangan nasional.", 1),
        ("Sistem pendidikan inklusif memastikan setiap anak, termasuk yang berkebutuhan khusus, mendapat akses belajar.", "Pendidikan inklusif menjamin bahwa semua anak termasuk anak berkebutuhan khusus memiliki akses terhadap pendidikan.", 1),
        ("Perubahan iklim mengancam ketahanan pangan global melalui pola cuaca ekstrem yang tidak menentu.", "Ancaman perubahan iklim terhadap ketahanan pangan global terwujud melalui pola cuaca ekstrem yang sulit diprediksi.", 1),
        ("Migrasi internasional meningkat akibat faktor ekonomi, konflik, dan perubahan lingkungan hidup.", "Peningkatan migrasi internasional didorong oleh faktor-faktor seperti ekonomi, konflik bersenjata, dan degradasi lingkungan.", 1),
        ("Kesehatan mental menjadi isu global yang membutuhkan perhatian serius dari pemerintah dan masyarakat.", "Isu kesehatan mental di tingkat global memerlukan perhatian yang serius dari pemerintah maupun masyarakat luas.", 1),
        ("Pengelolaan sampah yang buruk menyebabkan pencemaran lingkungan dan mengancam kesehatan masyarakat.", "Pencemaran lingkungan dan ancaman kesehatan masyarakat merupakan dampak nyata dari pengelolaan sampah yang tidak baik.", 1),
        # Label 0
        ("Fotosintesis adalah proses biokimia yang menggunakan cahaya matahari untuk menghasilkan glukosa.", "Resep membuat kue bolu membutuhkan tepung terigu, telur, gula pasir, dan mentega yang dikocok hingga mengembang.", 0),
        ("Hukum Newton menyatakan hubungan antara gaya dan percepatan sebuah benda bermassa.", "Budidaya ikan lele dapat dilakukan di kolam terpal dengan kepadatan tebar yang sesuai dan pakan rutin.", 0),
        ("Demokrasi membutuhkan partisipasi aktif warga negara dalam setiap proses politik.", "Teknik memasak sous vide menggunakan air panas dengan suhu terkontrol untuk menghasilkan masakan matang sempurna.", 0),
        ("Inflasi berpengaruh langsung terhadap daya beli masyarakat dan stabilitas ekonomi nasional.", "Olahraga lari maraton membutuhkan latihan fisik yang intensif selama beberapa bulan sebelum hari perlombaan.", 0),
        ("Kecerdasan buatan digunakan dalam berbagai bidang mulai dari kesehatan hingga transportasi massal.", "Budaya Bali dikenal sangat kaya dengan seni pertunjukan tradisional seperti tari kecak dan tari barong.", 0),
        ("Kurikulum 2013 menekankan pendekatan saintifik dan karakter dalam proses pembelajaran di sekolah.", "Cara merawat tanaman hias dalam ruangan meliputi penyiraman rutin, pemupukan, dan pencahayaan yang cukup.", 0),
        ("Pemanasan global menyebabkan mencairnya lapisan es di kutub utara dan selatan secara signifikan.", "Resep soto ayam tradisional menggunakan bumbu kuning dengan tambahan tauge, bihun, dan telur rebus.", 0),
        ("Revolusi digital telah mengubah hampir semua aspek kehidupan manusia modern saat ini.", "Teknik origami yang berasal dari Jepang melibatkan seni melipat kertas menjadi berbagai bentuk yang indah.", 0),
        ("Pancasila sebagai ideologi bangsa harus diamalkan dalam kehidupan bermasyarakat sehari-hari.", "Cara budidaya jamur tiram yang baik membutuhkan media serbuk kayu dan kelembapan ruangan yang selalu terjaga.", 0),
        ("Ekosistem hutan hujan tropis menyimpan sebagian besar keanekaragaman spesies di seluruh bumi.", "Panduan memilih laptop untuk kebutuhan gaming mencakup pertimbangan spesifikasi GPU, RAM, dan kapasitas baterai.", 0),
        ("Kebijakan moneter pemerintah berfokus pada pengendalian inflasi dan menjaga stabilitas nilai tukar rupiah.", "Cara membuat origami burung memerlukan kertas persegi dan teknik melipat yang tepat secara berurutan.", 0),
        ("Sistem imun tubuh manusia bekerja melawan infeksi bakteri dan virus yang masuk ke dalam tubuh.", "Panduan wisata ke Yogyakarta meliputi kunjungan ke Kraton, Borobudur, dan Malioboro yang terkenal.", 0),
        # Label 0 — topik sama
        ("Pembelajaran daring memudahkan akses pendidikan bagi siswa di daerah terpencil di Indonesia.", "Metode pembelajaran tatap muka memiliki keunggulan dalam membangun interaksi sosial langsung antar siswa.", 0),
        ("Ekonomi digital mendorong pertumbuhan UMKM melalui platform e-commerce yang mudah diakses.", "Kebijakan moneter pemerintah berfokus pada pengendalian inflasi dan menjaga stabilitas nilai tukar rupiah.", 0),
        ("Energi terbarukan menjadi solusi ramah lingkungan untuk mengurangi ketergantungan pada bahan bakar fosil.", "Penggunaan bahan bakar fosil masih mendominasi sektor industri berat di banyak negara berkembang saat ini.", 0),
        ("Penelitian ilmiah membutuhkan metodologi yang ketat, replikabilitas, dan objektivitas yang tinggi.", "Seni lukis ekspresionisme mengutamakan ekspresi emosi pelukis melalui goresan spontan dan warna berani.", 0),
        ("Perubahan iklim berdampak pada pola cuaca ekstrem dan mengancam ketahanan pangan secara global.", "Desain interior minimalis mengutamakan fungsi ruangan dengan dekorasi yang bersih dan tidak berlebihan.", 0),
        ("Perkembangan teknologi informasi mengubah cara masyarakat mengakses dan menyebarkan berita.", "Manajemen risiko investasi saham memerlukan diversifikasi portofolio dan analisis fundamental yang cermat.", 0),
        ("Sistem pendidikan tinggi di Indonesia diatur oleh Undang-Undang Nomor 12 Tahun 2012.", "Peraturan lalu intas mengharuskan pengendara sepeda motor menggunakan helm standar nasional saat berkendara.", 0),
        ("Vaksinasi massal merupakan strategi efektif untuk mencapai kekebalan kelompok terhadap penyakit menular.", "Program beasiswa pemerintah bertujuan meningkatkan akses pendidikan tinggi bagi masyarakat kurang mampu.", 0),
        ("Pertanian organik mengurangi penggunaan pestisida kimia dan menjaga keseimbangan ekosistem tanah.", "Arsitektur modern menggunakan bahan bangunan inovatif seperti beton bertulang dan kaca berlapis ganda.", 0),
    ]

    df_akademik = pd.DataFrame(akademik_pairs, columns=['sentence1', 'sentence2', 'label'])
    
    # Data Augmentation (8x)
    np.random.seed(42)
    aug_rows = []
    pos_pairs = df_akademik[df_akademik['label'] == 1]
    neg_pairs = df_akademik[df_akademik['label'] == 0]

    for _, row in pos_pairs.iterrows():
        s1, s2 = row['sentence1'], row['sentence2']
        w1, w2 = s1.split(), s2.split()
        aug_rows.append({'sentence1': s2, 'sentence2': s1, 'label': 1})
        if len(w1) > 4:
            aug_rows.append({'sentence1': s1, 'sentence2': ' '.join(w1[:max(4, int(len(w1)*0.9))]), 'label': 1})
        if len(w1) > 5:
            aug_rows.append({'sentence1': s1, 'sentence2': ' '.join(w1[:max(4, int(len(w1)*0.8))]), 'label': 1})
        if len(w1) > 6:
            aug_rows.append({'sentence1': ' '.join(w1[:max(4, int(len(w1)*0.7))]), 'sentence2': s2, 'label': 1})
        if len(w1) > 5:
            aug_rows.append({'sentence1': ' '.join(w1[1:]), 'sentence2': s2, 'label': 1})
        if len(w2) > 5:
            aug_rows.append({'sentence1': s1, 'sentence2': ' '.join(w2[:-1]), 'label': 1})
        if len(w2) > 5:
            aug_rows.append({'sentence1': ' '.join(w2[:max(4, int(len(w2)*0.8))]), 'sentence2': s1, 'label': 1})

    for _, row in neg_pairs.iterrows():
        aug_rows.append({'sentence1': row['sentence2'], 'sentence2': row['sentence1'], 'label': 0})

    df_aug = pd.concat([df_akademik, pd.DataFrame(aug_rows)], ignore_index=True)

    df_dev   = df_paws_dev.copy()
    df_test  = df_paws_test.copy()
    df_train = pd.concat([df_paws_train, df_aug], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nTotal Train: {len(df_train):,} | Dev: {len(df_dev):,} | Test: {len(df_test):,}")

    # Parallel Sastrawi preprocessing
    t0 = time.time()
    print("Preprocessing all texts in parallel...")
    all_sentences_train_1 = df_train['sentence1'].tolist()
    all_sentences_train_2 = df_train['sentence2'].tolist()
    all_sentences_dev_1 = df_dev['sentence1'].tolist()
    all_sentences_dev_2 = df_dev['sentence2'].tolist()
    all_sentences_test_1 = df_test['sentence1'].tolist()
    all_sentences_test_2 = df_test['sentence2'].tolist()

    all_texts_to_stem = (
        all_sentences_train_1 + all_sentences_train_2 +
        all_sentences_dev_1 + all_sentences_dev_2 +
        all_sentences_test_1 + all_sentences_test_2
    )
    
    stemmed_results = parallel_preprocess(all_texts_to_stem)
    
    # Assign back
    n_train = len(df_train)
    n_dev = len(df_dev)
    n_test = len(df_test)
    
    df_train['s1_clean'] = stemmed_results[0 : n_train]
    df_train['s2_clean'] = stemmed_results[n_train : 2*n_train]
    df_dev['s1_clean']   = stemmed_results[2*n_train : 2*n_train + n_dev]
    df_dev['s2_clean']   = stemmed_results[2*n_train + n_dev : 2*n_train + 2*n_dev]
    df_test['s1_clean']  = stemmed_results[2*n_train + 2*n_dev : 2*n_train + 2*n_dev + n_test]
    df_test['s2_clean']  = stemmed_results[2*n_train + 2*n_dev + n_test : 2*n_train + 2*n_dev + 2*n_test]
    
    print(f"Preprocessing completed in {time.time() - t0:.2f} seconds.")

    # Fit TF-IDF vectorizers
    print("Fitting TF-IDF Vectorizers...")
    all_clean_sents = pd.concat([df_train['s1_clean'], df_train['s2_clean']], ignore_index=True)

    vectorizer_word = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 4),
        min_df=2, max_features=100000, sublinear_tf=True
    )
    vectorizer_word.fit(all_clean_sents)

    vectorizer_char = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 5),
        min_df=3, max_features=60000, sublinear_tf=True
    )
    vectorizer_char.fit(all_clean_sents)

    print(f"Word vocab: {len(vectorizer_word.vocabulary_):,} | Char vocab: {len(vectorizer_char.vocabulary_):,}")

    # Feature Extraction
    print("Extracting features - Train set...")
    X_train = extract_features(df_train, 'train')
    print("Extracting features - Dev set...")
    X_dev   = extract_features(df_dev, 'dev')
    print("Extracting features - Test set...")
    X_test  = extract_features(df_test, 'test')

    y_train = df_train['label'].values
    y_dev   = df_dev['label'].values
    y_test  = df_test['label'].values

    # Train XGBoost
    print("Training XGBoost Classifier...")
    clf = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_prob_dev = clf.predict_proba(X_dev)[:, 1]
    
    # Find optimal threshold on dev set
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = 0
    threshold_optimal = 0.5
    for th in thresholds:
        y_pred_th = (y_prob_dev >= th).astype(int)
        f1 = f1_score(y_dev, y_pred_th)
        if f1 > best_f1:
            best_f1 = f1
            threshold_optimal = th
            
    print(f"Optimal Threshold on Dev Set: {threshold_optimal:.4f} (Best F1: {best_f1:.4f})")

    # Evaluate on test set
    y_prob_test = clf.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= threshold_optimal).astype(int)

    auc_test = roc_auc_score(y_test, y_prob_test)
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    rec_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)

    print("=" * 65)
    print("  EVALUASI MODEL A v8 OPTIMIZED")
    print("=" * 65)
    print(f"  AUC       : {auc_test:.4f}")
    print(f"  Accuracy  : {acc_test:.4f}")
    print(f"  Precision : {prec_test:.4f}")
    print(f"  Recall    : {rec_test:.4f}")
    print(f"  F1-Score  : {f1_test:.4f}")
    print("=" * 65)

    # Save artifacts
    print("Saving model artifacts...")
    os.makedirs('model_artifacts', exist_ok=True)
    joblib.dump(vectorizer_word, 'model_artifacts/tfidf_vectorizer.pkl')
    joblib.dump(vectorizer_char, 'model_artifacts/tfidf_vectorizer_char.pkl')
    joblib.dump(clf,             'model_artifacts/xgboost_classifier.pkl')

    config = {
        'version'          : 'v8',
        'model_type'       : 'XGBoost (optimized training)',
        'threshold_optimal': float(threshold_optimal),
        'feature_names'    : FEATURE_NAMES,
        'n_features'       : len(FEATURE_NAMES),
        'auc_test'         : float(auc_test),
        'accuracy_test'    : float(acc_test),
        'f1_test'          : float(f1_test),
        'train_size'       : len(df_train),
        'dev_size'         : len(df_dev),
        'test_size'        : len(df_test),
    }
    joblib.dump(config, 'model_artifacts/pipeline_config.pkl')
    print("Model artifacts saved successfully!")
