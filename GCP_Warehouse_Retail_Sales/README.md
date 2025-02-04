# Optimizing Sales and Inventory Management for a Beverage Distributor

## ADTA 5240: Harvesting, Storing, and Retrieving Data

### **Presented by:**
**Sri Sai Durga Katreddi**

---

## 📌 Introduction
This project focuses on optimizing **sales and inventory management** for a beverage distributor by analyzing historical sales data. The goal is to **predict future stock requirements** and enhance decision-making using data visualization and analytics tools.

---

## 📊 Data Lifecycle Overview

### 1️⃣ **Data Collection**
- Dataset sourced from [data.gov](https://catalog.data.gov/dataset/warehouse-and-retail-sales)
- Contains **307,646 rows** and **9 columns**:
  - `YEAR`, `MONTH`, `SUPPLIER`, `ITEM CODE`, `ITEM DESCRIPTION`, `ITEM TYPE`, `RETAIL SALES`, `RETAIL TRANSFERS`, `WAREHOUSE SALES`
- **Data Inconsistencies:**
  - `ITEM CODE` contained character values (e.g., "WC", "BC").
  - Replaced **"WC" → `20001`** and **"BC" → `20002`** for consistency.

### 2️⃣ **Data Storage**
- **Stored in BigQuery** (Google Cloud Platform)
- Steps:
  1. Enable BigQuery in **GCP Console**
  2. Upload dataset for processing

### 3️⃣ **Data Processing**
- **NULL values** found in `SUPPLIER`, `ITEM_TYPE`, `RETAIL_SALES`
- Handled using:
  - `SUPPLIER = "Unknown"`
  - `ITEM_TYPE = "0"`
  - `RETAIL_SALES = 0`
- Ensured **data integrity** before analysis

### 4️⃣ **Data Analysis**
- **SQL queries in BigQuery** to analyze:
  - Total sales by supplier
  - Retail trends & performance metrics
- **Spark-SQL vs Hive:**
  - **Spark-SQL** outperformed Hive by **50-87% faster execution time** 🚀

### 5️⃣ **Data Visualization**
- **Tableau/Power BI** used for insights:
  - Sales trends over time 📈
  - Supplier performance
  - Sales-to-warehouse ratios

---

## 🔍 Key Insights
1️⃣ **Sales peaked in 2020**, likely due to **pandemic-driven buying**. A decline in 2021 followed by recovery in 2022.  
2️⃣ **Top Suppliers:**
   - **E. & J. Gallo Winery**
   - **Diageo North America** (dominate the market)
3️⃣ **Liquor & Wine showed higher sales-to-transfer ratios**, indicating **stronger retail demand than beer**.

---

## 📈 Performance Metrics
- **Spark-SQL vs Hive Execution Times:**
  - Spark **50-87% faster** than Hive ⚡
  - **Preferred choice for large-scale analytics**

---

## 🚀 Conclusion & Next Steps
### **Conclusion**
By integrating **BigQuery, Spark-SQL, Hive, and Tableau**, we extracted **valuable insights** into sales and inventory trends, allowing for **data-driven decision-making**.

### **Next Steps for Data Science & Analyst Teams**
✅ Develop **Time-Series Models** for predicting **monthly sales trends** by supplier & product type 📊  
✅ Build **Predictive Models** for **demand forecasting** & **inventory optimization** 📉  
✅ Use **Classification Models** to **identify high-risk suppliers** & low-selling products 🎯  

---

## 📚 References
- [Google Cloud BigQuery](https://cloud.google.com/bigquery)
- [Google Cloud Hive with Dataproc](https://cloud.google.com/dataproc-metastore/docs/use-hive)
- [Google Cloud Spark with Dataproc](https://cloud.google.com/dataproc-metastore/docs/use-spark)
- [Extending SQL capabilities with Presto](https://cloud.google.com/blog/products/data-analytics/extending-the-sql-capabilities-of-your-cloud-dataproc-cluster-with-the-presto-optional-component)

---

