"""
Egyptian Arabic Knowledge Base
================================

This module provides Egyptian Arabic (العامية المصرية) knowledge entries
for the RAG engine. Each entry covers common questions users might ask
in Egyptian dialect, with multiple phrasings for the same concept.

Why Multiple Formats?
  Egyptian Arabic users express the same question in many ways:
    - "ايه هو الذكاء الاصطناعي" (What is AI?)
    - "الذكاء الاصطناعي يعني ايه" (What does AI mean?)
    - "اشرحلي الذكاء الاصطناعي" (Explain AI to me)

  By storing multiple phrasings, the TF-IDF retrieval has a higher chance
  of matching the user's specific wording, regardless of how they ask.

Topics Covered:
  - Artificial Intelligence & Machine Learning (ai_ml)
  - Programming fundamentals (programming)
  - Web Development (web_dev)
  - Data Science & Databases (data)
  - Career & Learning advice (general)
  - Python-specific (programming)
  - Common tech questions (general)
"""

# Each entry: (content_in_egyptian_arabic, topic_category)
# Multiple entries for the same concept = better retrieval coverage

EGYPTIAN_ARABIC_KNOWLEDGE = [

    # =========================================================================
    # ARTIFICIAL INTELLIGENCE & MACHINE LEARNING
    # =========================================================================

    # "What is AI?" — Multiple formats
    ("الذكاء الاصطناعي هو فرع من علوم الكمبيوتر بيخلي الاجهزة تقدر تفكر "
     "وتتعلم وتحل مشاكل زي البشر. بيشمل تعلم الالة والتعلم العميق ومعالجة اللغات",
     "ai_ml"),

    ("الذكاء الاصطناعي يعني ان الكمبيوتر يقدر يعمل حاجات كانت محتاجة ذكاء بشري "
     "زي فهم الكلام والصور واتخاذ القرارات والتعلم من البيانات",
     "ai_ml"),

    ("الـ AI ده ببساطة هو تعليم الكمبيوتر يفكر. بدل ما تقوله كل خطوة "
     "هو بيتعلم لوحده من الامثلة والبيانات اللي بتديهاله",
     "ai_ml"),

    # "What is Machine Learning?" — Multiple formats
    ("تعلم الالة هو جزء من الذكاء الاصطناعي. الفكرة انك بتدي الكمبيوتر بيانات كتير "
     "وهو بيتعلم منها الانماط لوحده من غير ما تبرمجه على كل حالة",
     "ai_ml"),

    ("الـ Machine Learning يعني ان الموديل بيتدرب على داتا ويتعلم يتوقع حاجات جديدة. "
     "فيه supervised learning و unsupervised learning و reinforcement learning",
     "ai_ml"),

    ("تعلم الالة بيشتغل ازاي: بتجمع بيانات وبتنضفها وبتختار موديل وبتدربه "
     "على البيانات وبعدين بتختبره. لو النتايج كويسة بتستخدمه في التطبيق بتاعك",
     "ai_ml"),

    # "What is Deep Learning?"
    ("التعلم العميق هو نوع متقدم من تعلم الالة بيستخدم شبكات عصبية كتير الطبقات. "
     "هو اللي ورا التعرف على الصور والترجمة الالية وتوليد النصوص زي ChatGPT",
     "ai_ml"),

    ("الـ Deep Learning بيستخدم neural networks عميقة فيها طبقات كتير. "
     "كل طبقة بتتعلم ميزة معينة. مثلا في الصور الطبقة الاولى بتتعلم الحواف "
     "والتانية الاشكال والتالتة الوشوش الكاملة",
     "ai_ml"),

    # "What is NLP?"
    ("معالجة اللغات الطبيعية NLP هي الفرع اللي بيخلي الكمبيوتر يفهم كلام البشر. "
     "بتستخدم في الشات بوت والترجمة وتحليل المشاعر وتلخيص النصوص",
     "ai_ml"),

    # "What are neural networks?"
    ("الشبكات العصبية هي نظام حوسبي مستوحى من المخ البشري. البيانات بتمر "
     "عبر طبقات من العقد المتصلة ببعض وكل اتصال له وزن بيتعدل اثناء التدريب",
     "ai_ml"),

    # "What is a transformer model?"
    ("الـ Transformer هو معمارية بتستخدم الـ attention mechanism عشان تفهم العلاقات "
     "بين الكلمات في الجملة. هو اساس موديلات زي GPT و BERT. ميزته انه بيفهم "
     "السياق البعيد في النص",
     "ai_ml"),

    # "What is overfitting?"
    ("الـ overfitting يعني ان الموديل حفظ بيانات التدريب بدل ما يتعلم الانماط العامة. "
     "بيبقى كويس جدا على بيانات التدريب بس وحش على بيانات جديدة. "
     "الحل: regularization او dropout او زيادة البيانات",
     "ai_ml"),

    # =========================================================================
    # PROGRAMMING FUNDAMENTALS
    # =========================================================================

    # "What is the best programming language?" — Multiple formats
    ("مفيش لغة برمجة افضل من التانية بشكل مطلق. كل لغة ليها استخداماتها. "
     "Python للذكاء الاصطناعي والداتا. JavaScript للويب. Java للاندرويد والانظمة الكبيرة. "
     "C++ للالعاب والانظمة. اختار على حسب المجال اللي عايز تشتغل فيه",
     "programming"),

    ("افضل لغة برمجة تبدأ بيها هي Python عشان سهلة وبسيطة ومطلوبة في السوق. "
     "بعدها ممكن تتعلم JavaScript للويب او Java للموبايل",
     "programming"),

    ("لو عايز تدخل مجال الذكاء الاصطناعي اتعلم Python. "
     "لو عايز تعمل مواقع اتعلم JavaScript. "
     "لو عايز تعمل تطبيقات موبايل اتعلم Kotlin او Flutter",
     "programming"),

    # "How do I start programming?" — Multiple formats
    ("عشان تبدأ تتعلم برمجة: اول حاجة اختار لغة مبدئيا Python افضل حاجة للمبتدئين. "
     "اتعلم الاساسيات زي المتغيرات والدوال والشروط واللوبات. "
     "بعدين ابدأ اعمل مشاريع صغيرة وكبر المشاريع تدريجيا",
     "programming"),

    ("لو مبتدئ في البرمجة ابدأ كده: اول حاجة Python عشان سهلة. "
     "اتعلم variables و loops و functions و if conditions. "
     "بعدين اعمل مشروع صغير زي calculator او to-do list. "
     "وبعدين ادخل على data structures و algorithms",
     "programming"),

    ("خطوات تعلم البرمجة من الصفر: "
     "اولا اتعلم اساسيات الكمبيوتر والمنطق. "
     "ثانيا اختار لغة واحدة بس واتقنها. "
     "ثالثا اعمل مشاريع عملية. "
     "رابعا اتعلم Git و GitHub. "
     "خامسا ابدأ تتخصص في مجال معين",
     "programming"),

    # "What is an algorithm?"
    ("الخوارزمية هي مجموعة خطوات محددة لحل مشكلة معينة. "
     "زي وصفة الطبخ بالظبط: خطوات واضحة بترتيب معين بتوصلك للنتيجة. "
     "امثلة: binary search و sorting algorithms و dynamic programming",
     "programming"),

    # "What is OOP?"
    ("البرمجة الكائنية OOP هي طريقة لتنظيم الكود في كائنات objects "
     "كل كائن فيه بيانات data ووظائف methods. "
     "المبادئ الاساسية: Encapsulation و Inheritance و Polymorphism و Abstraction",
     "programming"),

    # "What is Git?"
    ("Git هو نظام تحكم بالاصدارات بيتتبع التغييرات في الكود. "
     "بيخليك تحفظ نسخ commits وترجع لاي نسخة قديمة وتشتغل مع فريق بالبرانشات. "
     "GitHub هو موقع بتحط عليه المشاريع بتاعتك اونلاين",
     "programming"),

    # "What are data structures?"
    ("هياكل البيانات هي طرق لتنظيم وتخزين البيانات عشان نقدر نوصلها ونعدلها بسرعة. "
     "اهمها: Array و Linked List و Stack و Queue و Hash Table و Tree و Graph. "
     "كل واحدة ليها مميزات وعيوب في السرعة والذاكرة",
     "programming"),

    # =========================================================================
    # WEB DEVELOPMENT
    # =========================================================================

    # "How do I make an API in Python?" — Multiple formats
    ("عشان تعمل API في Python استخدم FastAPI او Flask. "
     "FastAPI احسن لانها اسرع وبتعمل documentation اوتوماتيك. "
     "الخطوات: install fastapi و uvicorn, اعمل ملف app.py, "
     "اكتب الـ endpoints بتاعتك, وشغل السيرفر بـ uvicorn app:app",
     "web_dev"),

    ("لعمل REST API في بايثون: "
     "اول حاجة pip install fastapi uvicorn. "
     "تاني حاجة اعمل ملف main.py واكتب فيه from fastapi import FastAPI وبعدين app = FastAPI(). "
     "تالت حاجة اعمل endpoints زي @app.get و @app.post. "
     "واخيرا شغلها بـ uvicorn main:app --reload",
     "web_dev"),

    # "What is an API?"
    ("الـ API يعني Application Programming Interface. "
     "ببساطة هو طريقة برنامجين يتكلموا مع بعض. "
     "زي الجرسون في المطعم: انت بتطلب حاجة والجرسون بيوصلها للمطبخ ويرجعلك بالاكل. "
     "الـ REST API بيستخدم HTTP methods زي GET و POST و PUT و DELETE",
     "web_dev"),

    # "What is HTML/CSS/JS?"
    ("HTML هو اللي بيحدد بنية صفحة الويب العناصر زي العناوين والفقرات والصور. "
     "CSS هو اللي بيتحكم في الشكل والالوان والتصميم. "
     "JavaScript هي اللي بتخلي الصفحة تفاعلية وديناميكية",
     "web_dev"),

    # "What is React?"
    ("React هي مكتبة JavaScript من Meta لبناء واجهات المستخدم. "
     "بتستخدم components قابلة لاعادة الاستخدام و virtual DOM للسرعة "
     "و hooks لادارة الـ state. هي الاكثر استخداما في سوق العمل حاليا",
     "web_dev"),

    # "What is frontend vs backend?"
    ("الـ Frontend هو الجزء اللي المستخدم بيشوفه ويتفاعل معاه في المتصفح "
     "HTML و CSS و JavaScript. الـ Backend هو الجزء اللي بيشتغل على السيرفر "
     "بيتعامل مع قواعد البيانات والمنطق والامان. الاتنين مع بعض بيعملوا موقع كامل",
     "web_dev"),

    # "What is deployment?"
    ("الـ Deployment يعني رفع المشروع بتاعك على سيرفر عشان الناس تقدر تستخدمه. "
     "ممكن تستخدم Render او Railway او Vercel للمشاريع الصغيرة مجانا. "
     "او AWS و Google Cloud للمشاريع الكبيرة. Docker بيساعدك تعمل حاوية "
     "بتشتغل بنفس الطريقة في اي مكان",
     "web_dev"),

    # =========================================================================
    # DATA SCIENCE & DATABASES
    # =========================================================================

    # "What is a database?"
    ("قاعدة البيانات هي مكان منظم لتخزين البيانات. "
     "فيه نوعين: SQL زي PostgreSQL و MySQL بتستخدم جداول وعلاقات. "
     "و NoSQL زي MongoDB بتستخدم documents مرنة. "
     "SQL افضل لما تحتاج علاقات بين البيانات و NoSQL افضل للمرونة والسرعة",
     "data"),

    # "What is SQL?"
    ("SQL هي لغة بتتكلم بيها مع قواعد البيانات. بتقدر تعمل: "
     "SELECT لجلب بيانات, INSERT لاضافة, UPDATE للتعديل, DELETE للحذف. "
     "JOIN لربط جداول مع بعض. هي لغة اساسية لاي مبرمج لازم يتعلمها",
     "data"),

    # "What is Data Science?"
    ("علم البيانات هو مجال بيجمع بين البرمجة والاحصاء والمعرفة بالمجال "
     "عشان يستخرج معلومات مفيدة من البيانات. الادوات الاساسية: "
     "Python و pandas للتحليل و matplotlib للرسم و scikit-learn للموديلات",
     "data"),

    # "What is pandas?"
    ("pandas هي مكتبة Python للتعامل مع البيانات الجدولية. "
     "بتخليك تقرأ ملفات CSV و Excel وتنظف البيانات وتعمل تحليلات "
     "وتجميع وفلترة بسهولة. هي اساسية لاي حد شغال في Data Science",
     "data"),

    # =========================================================================
    # PYTHON SPECIFIC
    # =========================================================================

    # "What is Python?"
    ("Python لغة برمجة سهلة وقوية جدا. بتستخدم في: "
     "الذكاء الاصطناعي بـ PyTorch و TensorFlow. "
     "تطوير الويب بـ Django و FastAPI. "
     "تحليل البيانات بـ pandas و NumPy. "
     "الاتمتة والسكريبتات. هي من اكثر اللغات طلبا في سوق العمل",
     "programming"),

    # "How to install Python?"
    ("عشان تنزل Python على جهازك: "
     "روح على python.org ونزل اخر اصدار. "
     "اثناء التنصيب متنساش تعلم على Add to PATH. "
     "بعدها افتح Terminal واكتب python --version عشان تتاكد انها شغالة. "
     "استخدم pip install عشان تنزل مكتبات",
     "programming"),

    # "What is pip?"
    ("pip هو مدير الحزم بتاع Python. بيخليك تنزل وتدير المكتبات بسهولة. "
     "pip install package_name عشان تنزل مكتبة. "
     "pip freeze عشان تشوف المكتبات المنزلة. "
     "pip install -r requirements.txt عشان تنزل كل متطلبات المشروع مرة واحدة",
     "programming"),

    # "What is a virtual environment?"
    ("الـ Virtual Environment هو بيئة معزولة لكل مشروع Python. "
     "بيخليك كل مشروع يبقى له مكتبات خاصة بيه من غير ما يأثر على المشاريع التانية. "
     "اعمله بـ python -m venv myenv وشغله بـ source myenv/bin/activate",
     "programming"),

    # =========================================================================
    # CAREER & LEARNING ADVICE
    # =========================================================================

    # "Where can I learn programming?" — Multiple formats
    ("اماكن تتعلم منها برمجة: "
     "مجانا: freeCodeCamp و Codecademy و CS50 من هارفارد و YouTube. "
     "بالعربي: قناة Elzero Web School و Codezilla و Abdullah Eid. "
     "بفلوس: Udemy و Coursera و المعسكرات التدريبية. "
     "الاهم انك تطبق وتعمل مشاريع مش بس تتفرج",
     "general"),

    ("لو عايز تتعلم برمجة مجانا بالعربي: "
     "Elzero Web School على يوتيوب عنده كورسات HTML و CSS و JavaScript و Python. "
     "Codezilla عنده محتوى عربي ممتاز. "
     "وفيه منصة harmash.com فيها شروحات بالعربي. "
     "ابدأ بمشاريع صغيرة وكل يوم اكتب كود",
     "general"),

    # "How to get a programming job?"
    ("عشان تلاقي شغل برمجة: "
     "اول حاجة ابني portfolio فيه 3-5 مشاريع قوية على GitHub. "
     "تاني حاجة اتعلم المطلوب في السوق اللي انت فيه. "
     "تالت حاجة جهز CV كويس وحط لينك GitHub. "
     "رابع حاجة قدم على LinkedIn و Wuzzuf و Indeed. "
     "خامس حاجة تدرب على المقابلات التقنية",
     "general"),

    # "How to prepare for coding interviews?"
    ("عشان تجهز لمقابلات البرمجة: "
     "اول حاجة اتعلم Data Structures و Algorithms كويس. "
     "تاني حاجة حل مسائل على LeetCode كل يوم على الاقل مسألة. "
     "تالت حاجة اتعلم System Design لو خبرتك فوق السنتين. "
     "رابع حاجة اعمل mock interviews مع حد. "
     "والاهم انك تفكر بصوت عالي وتشرح طريقة تفكيرك",
     "general"),

    # "Freelancing advice"
    ("لو عايز تشتغل فريلانس في البرمجة: "
     "ابدأ على Upwork او Freelancer او Mostaql. "
     "اعمل بروفايل قوي وحط مشاريعك السابقة. "
     "ابدأ باسعار معقولة عشان تجمع تقييمات. "
     "تخصص في حاجة معينة بدل ما تعمل كل حاجة. "
     "والتزم بالمواعيد عشان تبني سمعة كويسة",
     "general"),

    # =========================================================================
    # COMMON TECH QUESTIONS IN EGYPTIAN ARABIC
    # =========================================================================

    # "What is the difference between Python and JavaScript?"
    ("الفرق بين Python و JavaScript: "
     "Python: بتستخدم للذكاء الاصطناعي والداتا والباك اند. بسيطة وسهلة القراءة. "
     "JavaScript: بتستخدم للويب فرونت اند وباك اند بـ Node.js. بتشتغل في المتصفح. "
     "لو عايز AI اتعلم Python. لو عايز ويب اتعلم JavaScript. الاتنين مطلوبين جدا",
     "programming"),

    # "What is Docker?"
    ("Docker هو اداة بتحط التطبيق بتاعك في حاوية container فيها كل حاجة محتاجها. "
     "الميزة ان التطبيق بيشتغل بنفس الطريقة على اي جهاز. "
     "بدل ما تقول الكود بيشتغل عندي بس مش عندك, Docker بيحل المشكلة دي",
     "programming"),

    # "What is Linux?"
    ("Linux هو نظام تشغيل مفتوح المصدر بيستخدم في السيرفرات والبرمجة كتير. "
     "معظم السيرفرات في العالم بتشغل Linux. كمبرمج لازم تتعلم اساسيات "
     "الـ Terminal والاوامر زي cd و ls و mkdir و grep و chmod",
     "programming"),

    # "What is Cloud Computing?"
    ("الحوسبة السحابية يعني انك بتستأجر موارد كمبيوتر من الانترنت "
     "بدل ما تشتري سيرفرات خاصة. اشهر المنصات AWS و Google Cloud و Azure. "
     "بتقدر تأجر سيرفرات وقواعد بيانات وتخزين وكلها بتتحاسب بالاستخدام",
     "general"),

    # "What is Cybersecurity?"
    ("الامن السيبراني هو حماية الانظمة والشبكات والبيانات من الهجمات. "
     "بيشمل: حماية التطبيقات من SQL Injection و XSS, "
     "تشفير البيانات, ادارة الصلاحيات, واكتشاف الثغرات. "
     "مجال مطلوب جدا ومرتباته عالية",
     "general"),

    # "What is an IDE?"
    ("الـ IDE هو بيئة تطوير متكاملة. ببساطة هو البرنامج اللي بتكتب فيه الكود. "
     "اشهرهم VS Code وهو مجاني وخفيف وفيه اضافات كتير. "
     "PyCharm لـ Python. IntelliJ لـ Java. "
     "كل IDE بيوفر تلوين الكود واكمال تلقائي وتشغيل وديباجينج",
     "programming"),

    # "What is version control?"
    ("التحكم بالاصدارات هو نظام بيتتبع كل تغيير في الكود. "
     "بيخليك: ترجع لاي نسخة قديمة, تشتغل مع فريق من غير تضارب, "
     "تعمل branches لميزات جديدة. Git هو الاشهر و GitHub هو المنصة الاشهر. "
     "لازم كل مبرمج يتعلمه من اول يوم",
     "programming"),

    # "What is testing?"
    ("الاختبارات في البرمجة بتضمن ان الكود بيشتغل صح. فيه انواع: "
     "Unit Testing بتختبر كل دالة لوحدها. "
     "Integration Testing بتختبر ان الاجزاء بتشتغل مع بعض. "
     "End-to-End Testing بتختبر التطبيق كله من وجهة نظر المستخدم. "
     "في Python بنستخدم pytest",
     "programming"),

    # "Tips for beginners"
    ("نصائح للمبتدئين في البرمجة: "
     "اولا متستعجلش, البرمجة محتاجة صبر. "
     "ثانيا اكتب كود كل يوم حتى لو 30 دقيقة. "
     "ثالثا الغلط طبيعي, الـ errors مش عيب دي جزء من التعلم. "
     "رابعا اعمل مشاريع عملية مش بس تتفرج على كورسات. "
     "خامسا انضم لمجتمعات مبرمجين واسأل واتعلم",
     "general"),
]
