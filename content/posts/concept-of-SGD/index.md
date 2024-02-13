---
title:  "مفهوم گرادیان نزولی تصادفی"
date: 2024-02-10T02:32:46+03:30
tags: ["گرادیان نزولی", "هوش مصنوعی", "یادگیری عمیق", "SGD", "مفاهیم"]
author: "سعید حسنی"
showToc: false
TocOpen: true
draft: false
hidemeta: false
comments: false
description: "گرادیان نزولی تصادفی"
hideSummary: false
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: false
ShowRssButtonInSectionTermList: true
UseHugoToc: true
cover:
    image: "https://raw.githubusercontent.com/hasanisaeed/hasanisaeed.github.io/main/content/posts/concept-of-SGD/images/sgd.webp"
    alt: "SGD"
    caption: "این عکس با هوش مصنوعی تولید شده :)" 
    relative: true
    hidden: false
# editPost:
#     URL: "https://github.com/hasanisaeed/hasanisaeed.github.io/blob/main/content"
#     Text: "متن رو ویرایش کن 🤗"
#     appendFilePath: true # to append file path to Edit link
---
![SGD](https://raw.githubusercontent.com/hasanisaeed/hasanisaeed.github.io/main/content/posts/concept-of-SGD/images/sgd.webp#center)

### قسمت اول: درک مفاهیم پایه
#### گرادیان نزولی چیه؟
تصور کنید که شما بالا یه کوه قرار گرفتین و هدف تون این هست که میخواین به پایین ترین نقطه دست پیدا کنین. اینجاست که:
> گرادیان نزولی کمکتون میکنه تا بهترین مسیر به سمت پایین رو پیدا کنین.

زیبایی گرادیان نزولی سادگی و ظرافتش هست. توی مثال کوه مون مفهوماینه که شما با یه نقطه تصادفی روی کوه سعی میکنین که از کوه بیان پایین. در هر مرحله شما به اطراف نگاه میکنین و همش سعی میکنین که تندترین شیب (گرادیان) انتخاب کنین. این کار رو بارها و بارها تکرار میکنین. حالا چقدر؟ بهش میرسیم.

توی گرادیان نزولی هر قدم که به پایین برمیدارین، به اون قدم تون میگن اصلاحا نرخ یادگیری یا همون `learning rate`. 
> احتمالا برات سوال پیش اومده که من قدم بلند بردارم یا قدمهای کوچیک. چیزایی که الان به ذهنت میاد مفهومریاضی داره که عجله نکنی و خسته نشی برات بیشتر باز میکنم.

#### مفهوم 'تصادفی' در گرادیان نزولی تصادفی
فرض کن بالا کوه مه غلیظ گرفته و تو نمیتونی همه جا رو به خوبی ببینی(مسیله هامون توی دنیای واقعی اینجوری هستن).احتمالا قدمهای کوچیک و *تصادفی* برمیداری؟ و سعی میکنی بارها و بارها این کار رو تکرار کنی. واسه همینه که یه مسیر زیگ زاگ مانند داری همش طی میکنی!
> پس اومدیم *تصادفی* قدم برداشتیم و هر بار حواس مون هست که قدم بعدی رو محتاط تر برداریم. این همون مفهوم تصادفی بودن هست.

### قسمت دوم: ریاضیات پشت پرده SGD
#### گامهای گرادیان
- قدم اول: مقداردهی اولیه
ابتدا، پارامترها (وزن‌ها) مدل خود را مقداردهی اولیه می‌کنیم. این کار می‌تونه به صورت تصادفی یا با استفاده از روش دیگری برای مقداردهی اولیه انجام بشه. نقطه شروع برای SGD حیاتی است زیرا بر مسیر رسید به پیدا کردن نقطه بهینه تاثیر میذاره. 
- قدم دوم: انتخاب تصادفی
در هر تکرار از فرآیند آموزش، SGD به صورت تصادفی یک نقطه داده (یا یک دسته کوچک از نقاط داده) را از کل دیتاست انتخاب می‌کنه. اینکه که گفتیم تصادفی تصادفی، بخاطر این قسمت از الگورتیم بود! 
- قدم سوم: محاسبه گرادیان یا همون شیب
گرادیان تابع زیان را محاسبه کنین، اما فقط برای نقاط داده‌ای که به صورت تصادفی انتخاب شده‌اند. گرادیان یک بردار است که در جهت افزایش تندترین تابع زیان اشاره می‌کند. در مفهوم SGD، یعنی چگونه پارامترها رو برای دقیق‌تر کردن مدل برای آن نقطه داده خاص تنظیم کنیم. 

$$g_t = \nabla_\theta f_t(\theta_{t-1})$$

- قدم چهارم: آپدیت پارامترها
#### درک مفهوم نرخ یادگیری
فهمیدن و انتخاب درست پارامتر نرخ یادگیری، نقش زیادی در  کارامدی مدل SGD داره.

**اگر نرخ یادگیری بزرگ انتخاب بشه چه اتفاقی  میفته؟** فرض کن با گامهای بلند بخوایم به جلو حرکت کنیم و اون نقطه بهینه رو پیدا کنیم. ممکنه فک کنین که خب سریعتر به پایین ترین نقطه میرسم امااین کار باعث میشه مثلا یه گام جلوتر شما نقطه بهینه (پایین ترین نقطه) باشه و شما از روش بپری و ممکنه نقطه بهینه زیر پات، یه کم جلوتر باشه و توی هی عقب و جلو بپری.

**حالا اگه نرخ یادگیری رو کم بگیری چه اتفاقی میفته؟** فک کنم واضح باشه؟ نه؟ فرض کن افتادی توی یه گودی و  قدمهات خیلی کوچیک هست. حالا هرچقدر به چپ و راست میری بازم توی گودی هستی! و مشکل بعدی هم این هست که خیلی کند داری به سمت هدفت که همون پیدا کردن پایین نقطه هست میرسی.

**خب پس چجوری قدمها رو بردارم؟**
واسه این کار توصیه شده که نه قدمهای بلند بردارین و نه قدمی خیلی کوچیک. نرخ یادگیری رو از طریق سعی و خطا بدست میارن و میان در طول زمان مقدارش رو کم میکنن.

> یه سری استراتژی ها برای این کار مطرح شده که سرچ کنین دست تون میاد. مثلا بیایم به صورت نمایی نرخ رو کم کنیم یا بیان از یه سری متدها که الگوریتمهایی مثلا `Adam` ازش استفاده میکنن، استفاده کنیم و ...

### قسمت سوم: بریم سمت کدنویسی SGD
خب اول میام کد پایتون رو به طول کامل اینجا میذارم:

<details>
<summary> کد پیاده سازی شده با پایتون </summary>

```python
class SGDRegressor:
    def __init__(self, learning_rate=0.01, epochs=100, batch_size=1, reg=None, reg_param=0.0):
        """
        Constructor for the SGDRegressor.

        Parameters:
        learning_rate (float): The step size used in each update.
        epochs (int): Number of passes over the training dataset.
        batch_size (int): Number of samples to be used in each batch.
        reg (str): Type of regularization ('l1' or 'l2'); None if no regularization.
        reg_param (float): Regularization parameter.

        The weights and bias are initialized as None and will be set during the fit method.
        """
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        self.reg_param = reg_param
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the SGDRegressor to the training data.

        Parameters:
        X (numpy.ndarray): Training data, shape (m_samples, n_features).
        y (numpy.ndarray): Target values, shape (m_samples,).

        This method initializes the weights and bias, and then updates them over a number of epochs.
        """
        m, n = X.shape  # m is number of samples, n is number of features
        self.weights = np.zeros(n)
        self.bias = 0

        for _ in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, m, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                gradient_w = -2 * np.dot(X_batch.T, (y_batch - np.dot(X_batch, self.weights) - self.bias)) / self.batch_size
                gradient_b = -2 * np.sum(y_batch - np.dot(X_batch, self.weights) - self.bias) / self.batch_size

                if self.reg == 'l1':
                    gradient_w += self.reg_param * np.sign(self.weights)
                elif self.reg == 'l2':
                    gradient_w += self.reg_param * self.weights

                self.weights -= self.learning_rate * gradient_w
                self.bias -= self.learning_rate * gradient_b

    def predict(self, X):
        """
        Predicts the target values using the linear model.

        Parameters:
        X (numpy.ndarray): Data for which to predict target values.

        Returns:
        numpy.ndarray: Predicted target values.
        """
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, X, y):
        """
        Computes the loss of the model.

        Parameters:
        X (numpy.ndarray): The input data.
        y (numpy.ndarray): The true target values.

        Returns:
        float: The computed loss value.
        """
        return (np.mean((y - self.predict(X)) ** 2) + self._get_regularization_loss()) ** 0.5

    def _get_regularization_loss(self):
        """
        Computes the regularization loss based on the regularization type.

        Returns:
        float: The regularization loss.
        """
        if self.reg == 'l1':
            return self.reg_param * np.sum(np.abs(self.weights))
        elif self.reg == 'l2':
            return self.reg_param * np.sum(self.weights ** 2)
        else:
            return 0

    def get_weights(self):
        """
        Returns the weights of the model.

        Returns:
        numpy.ndarray: The weights of the linear model.
        """
        return self.weights
```

<details>
</details>

<summary> کد پیاده سازی شده با C </summary>
</details>

#### پیاده‌سازی SGD در مدل‌های یادگیری ماشین
#### پیاده‌سازی SGD در Sci-kit Learn و Tensorflow

### قسمت چهارم: مزایا و چالشهای SGD
#### چرا SGD رو انتخاب کنیم؟
#### غلبه بر چالش‌های مرتبط با SGD

###  قسمت پنجم: یه کم فراتر از مفاهیم بریم

#### انواع مختلف SGD
#### آینده‌ی SGD

### نتیجه گیری
