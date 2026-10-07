# Makine Öğrenmesi Yöntemleri Kullanılarak Endüstriyel Ekipmanlarda Arıza Tahmini

## Proje Hakkında

Bu projede, endüstriyel ekipmanların çalışma verileri kullanılarak makine arızalarının tahmin edilmesi amaçlanmaktadır.

Makinelerin çalışma sırasında oluşturduğu sıcaklık, dönme hızı, tork ve takım aşınması gibi veriler incelenerek, makinenin arıza yapıp yapmayacağını tahmin eden makine öğrenmesi modelleri geliştirilmesi planlanmaktadır.

Proje kapsamında veri setinin incelenmesi, veri analizi ve ön işleme işlemlerinin gerçekleştirilmesi, farklı makine öğrenmesi modellerinin uygulanması ve elde edilen sonuçların karşılaştırılması hedeflenmektedir.

## Problem Tanımı

Projede temel olarak şu soruya cevap aranacaktır:

> Makinenin mevcut çalışma verilerine bakarak arıza yapıp yapmayacağını tahmin edebilir miyiz?

Bu nedenle `Machine Failure` değişkeni hedef değişken olarak kullanılacaktır.

- `0` → Makine arızası yok
- `1` → Makine arızası var

Proje bir **ikili sınıflandırma problemi** olarak ele alınacaktır.

## Veri Seti

Projede **AI4I 2020 Predictive Maintenance Dataset** veri setinin kullanılması planlanmaktadır.

Veri seti **UCI Machine Learning Repository** üzerinden alınmıştır. Veri setinde endüstriyel makinelerin çalışma durumlarını gösteren farklı değişkenler bulunmaktadır.

Başlıca değişkenler:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Type

Veri seti öncelikle Python kullanılarak incelenecek; değişkenler, veri tipleri, eksik veriler ve verilerin dağılımları kontrol edilecektir.

## Veri Analizi ve Ön İşleme

Veri setindeki değişkenlerin makine arızası ile ilişkisi grafikler ve temel veri analizi yöntemleri kullanılarak incelenecektir.

Gerekli veri temizleme ve ön işleme işlemlerinden sonra modelde kullanılacak değişkenler belirlenecek ve veriler eğitim/test olarak ayrılacaktır.

## Kullanılacak Makine Öğrenmesi Yöntemleri

Projede farklı sınıflandırma yöntemlerinin uygulanması ve sonuçlarının karşılaştırılması planlanmaktadır.

İlk aşamada:

- Logistic Regression
- Decision Tree
- Random Forest

modellerinin kullanılması düşünülmektedir.

Proje ilerledikçe kullanılan veri ve elde edilen sonuçlara göre farklı modellerin eklenmesi değerlendirilebilir.

## Model Değerlendirme

Modellerin performansını değerlendirmek için aşağıdaki ölçütlerin kullanılması planlanmaktadır:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Özellikle arızalı makinelerin ne kadarının doğru şekilde tespit edildiğini görmek için **Recall** değeri incelenecektir.

Farklı modellerin sonuçları karşılaştırılarak hangi modelin ve hangi değişkenlerin tahmin sürecinde daha etkili olduğu değerlendirilecektir.

## Proje Akışı

```text
Veri Setini İnceleme
        ↓
Veri Temizleme ve Ön İşleme
        ↓
Veri Analizi
        ↓
Kullanılacak Değişkenleri Belirleme
        ↓
Eğitim / Test Verisi Oluşturma
        ↓
Makine Öğrenmesi Modelleri
        ↓
Arıza Tahmini
        ↓
Model Sonuçlarını Değerlendirme
