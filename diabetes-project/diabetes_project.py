


#Hasta tahlil veri seti kullanarak şeker hastalığı tahmin etme yapay zeka projesi
# merhaba bu projede bir hastanedeki hastalara ait gerçek tahlil verilerini kullanarak şeker hastalığı tahmini yapan yazay zeka kodluyorum.
#Yapay zeka modeli olarak, Makine öğrenmesi modellerinden KNN modelini Pythonda şeker hastalığı veri setinde kullanıyorum. 
#Bu veri seti Hindistandaki bir hastahaneden alınmış gerçek veriler, 768 kadın hastanın kan tahlili ve bazı istatistik bilgileri yüklü.


#Proje sonunda makine öğrenimi modelimiz sadece bu verilere değil tüm verere bakarak analiz yapmalı, 
#bu sadece yaş ve glukoz istatistiklerinden elde ettiiklerimiz,yapay zekanın en başarılı tahmini yapması için tüm istatistikleri sırayla karşılaştırmalıyiz.

#Bir sonraki analizimde veri setinde y eksenin outcome values, yani hasta mı değil mi (0/1)
#X değeri ise geri kalan değerler.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# Outcome = 1 Diabet/Şeker Hastası
# Outcome = 0 Sağlıklı
data = pd.read_csv("diabetes.csv")
data.head()

#Grafik çizdirmek için dataframe'de  sağlıklı insanları 0, şeker hastalarini 1 olarak kodluyorum.  
#X eksenini yaş, y eksenini glukoz olan scatter olarak görselleştirmesini istiyorum. 
#Kırmızı noktalar hasta, yeşil noktalar sağlıklı insanları temsil ediyor ve alpha ile bu renklerin koyuluğunu belirliliyorum.

seker_hastalari = data[data.Outcome == 1]
saglikli_insanlar = data[data.Outcome == 0]




#Proje sonunda makine öğrenimi modelimiz sadece bu verilere değil tüm verere bakarak analiz yapmalı, 
#bu sadece yaş ve glukoz istatistiklerinden elde ettiiklerimiz,yapay zekanın en başarılı tahmini yapması için tüm istatistikleri sırayla karşılaştırmalıyiz.


# Şimdilik sadece gloucose'a bakarak örnek bir çizim yapalım:
# Programımızın sonunda makine öğrenme modelimiz sadece glikoza değil tüm diğer verilere bakarak bir tahmin yapacaktır..
plt.scatter(saglikli_insanlar.Age, saglikli_insanlar.Glucose, color="green", label="sağlıklı", alpha = 0.4)
plt.scatter(seker_hastalari.Age, seker_hastalari.Glucose, color="red", label="diabet hastası", alpha = 0.4)
plt.xlabel("Age")
plt.ylabel("Glucose")
plt.legend()
plt.show()


#Bir sonraki analizimde veri setinde y eksenin outcome values, yani hasta mı değil mi (0/1)
#X değeri ise geri kalan değerler.
# x ve y eksenlerini belirleyelim
y = data.Outcome.values
x_ham_veri = data.drop(["Outcome"],axis=1)   


# Outcome sütununu(dependent variable) çıkarıp sadece independent variables bırakıyorum
# Çüknü KNN algoritması x değerleri içerisinde gruplandırma yapacak..


#KNN modelinde noktaların uzaklığının doğru hesaplanması için, değerlerin hepsi aynı etkiye sahip olmalı tüm değişkenlerin, istatiksel değerlerin eşit etkide olması gerekiyor. 
#Bunun için Hepsini 0-1 arasında belirlemem gerekiyor, normalizasyon işlemi yapıyorum. 
#Her bir kayıt için Ham veriden minimum ham veriyi çıkarıp o alandaki Max aralığa bölüyorum. X ekseninde ki değerlerin hepsi 0-1 aralığında oluyor.

# normalization yapıyorum - x_ham_veri içerisindeki değerleri sadece 0 ve 1 arasında olacak şekilde hepsini güncelliyorum
# Eğer bu şekilde normalization yapmazsam yüksek rakamlar küçük rakamları ezer ve KNN algoritmasını yanıltabilir!
x = (x_ham_veri - np.min(x_ham_veri))/(np.max(x_ham_veri)-np.min(x_ham_veri))

# önce
print("Normalization öncesi ham veriler:\n")
print(x_ham_veri.head())


# sonra 
print("\n\n\nNormalization sonrası yapay zekaya eğitim için vereceğimiz veriler:\n")
print(x.head())
    

#Veri setimi Train ve test olmak üzere ikiye ayırıyorum. Train data, sistemin sağlıklı insan ile hasta insanı ayırt etmeaini öğrenmesi için kullanacağım. 
#Test datayi ise makine öğrenme modelimi doğru bir şekilde hasta ve sağlıklı insanları ayırt edebiliyor mu diye test etmek için kullanacağım. 
#Ve yüzde başarısını ölçeceğim.

# train datamız ile test datamızı ayırıyorum
# train datamız sistemin sağlıklı insan ile hasta insanı ayırt etmesini öğrenmek için kullanılacak
# test datamız ise bakalım makine öğrenme modelimiz doğru bir şekilde hasta ve sağlıklı insanları ayırt edebiliyor mu diye 
# test etmek için kullanılacak...
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1,random_state=1)



# knn modelimizi oluşturuyorum.
knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbors = k
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print("K=3 için Test verilerimizin doğrulama testi sonucu ", knn.score(x_test, y_test))



# k kaç olmalı ?
# en iyi k değerini belirleyelim..
sayac = 1
for k in range(1,11):
    knn_yeni = KNeighborsClassifier(n_neighbors = k)
    knn_yeni.fit(x_train,y_train)
    print(sayac, "  ", "Doğruluk oranı: %", knn_yeni.score(x_test,y_test)*100)
    sayac += 1
    


#En yakın komşuyu hesaplama üzerine olan bu öğrenme modelinde k=3 seçtiğimde tahmin başarı oranı 0.78 oluyor. 
#K 1-10 arası değerler aldığında en yüksek başarıyı k = 7 aldığımda veriyor.
#İlk başta elimdeki 768 verinin %20 sini test için kullanmıştım %80 ini train için kullanmıştım ama bu oranı değiştirerek başarı oranının nasıl değişeceğine bakıyorum. 
#Bu sefer k =6 aldığımda daha başarılı oluyor.



#Bu projede kullandığım yapay zeka algoritması,K- nearest neighbours (KNN) modeli : 
#Daha önce elde edilen noktalara bakarak yeni bir nokta geldiği zaman onun hangi grupta olduğunu bulmak için kullanılır. 
#Gruplandırma, sınıflandırma mantığına dayanır. En yakındaki K tane komşu nokta bulunur. En çok hangi gruptan komşu nokta varsa yeni gelen nokta o gruptan olduğu kabul edilir.
#K= 3 dersek en yakın 3 noktaya bakarak bu gruplandırmayi yapar.
#Bu yapay zeka algoritması (KNN) yeni noktanın hangi grupta olduğunu bulurken Euclidean distance hesaplamasını kullanır.
#(a,b) = √(a1-b1)^2 + √(a2-b2)^2





# Yeni bir hasta tahmini için:
from sklearn.preprocessing import MinMaxScaler
 
# normalization yapıyoruz - daha hızlı normalization yapabilmek için MinMax  scaler kullandık...
sc = MinMaxScaler()
sc.fit_transform(x_ham_veri)
 
new_prediction = knn.predict(sc.transform(np.array([[6,148,72,35,0,33.6,0.627,50]])))
new_prediction[0]



