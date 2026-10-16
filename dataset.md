# Veri Seti Hakkında

Projede AI4I 2020 Predictive Maintenance Dataset veri setini kullanıyorum.

Veri seti 10.000 kayıt ve 14 değişkenden oluşuyor. Her kayıt bir makinenin
çalışma durumuyla ilgili bilgileri içeriyor.

## İncelediğim Değişkenler

- **UDI:** Kayıt için kullanılan benzersiz kimlik.
- **Product ID:** Ürün kimliği ve ürün tipi bilgisi.
- **Type:** Ürünün kalite seviyesini gösteriyor (L, M, H).
- **Air temperature [K]:** Hava sıcaklığı.
- **Process temperature [K]:** İşlem sıcaklığı.
- **Rotational speed [rpm]:** Dönme hızı.
- **Torque [Nm]:** Tork değeri.
- **Tool wear [min]:** Takımın kullanım süresi.
- **Machine failure:** Makinenin arıza yapıp yapmadığını gösteren hedef değişken.

## Machine Failure

`Machine failure` değişkeni makinenin arıza durumunu gösteriyor.

- `0` → Arıza yok
- `1` → Arıza var

Bu değişkeni makine öğrenmesi modelinde tahmin edilmesi gereken hedef
değişken olarak kullanmayı planlıyorum.

## Arıza Türleri

Veri setinde makine arızasının oluşmasına neden olabilecek beş farklı
arıza türü bulunuyor:

- **TWF:** Tool Wear Failure
- **HDF:** Heat Dissipation Failure
- **PWF:** Power Failure
- **OSF:** Overstrain Failure
- **RNF:** Random Failure

Bu arıza türlerinden en az biri gerçekleştiğinde `Machine failure`
değeri `1` olarak belirtiliyor.

## Şu An Yaptığım Çalışma

İlk olarak veri setini Python ve Pandas kullanarak okumaya ve yapısını
anlamaya çalışıyorum.

Veri setindeki satır ve sütunları, değişkenleri ve `Machine failure`
değerlerini inceliyorum. Ayrıca arızalı kayıtların sayısını bulmak ve
arızalı makinelerin diğer özelliklerini incelemek için filtreleme
işlemleri yapıyorum.

Bu incelemelerden sonra veriyi makine öğrenmesi modeli için hazırlamaya
geçeceğim.
