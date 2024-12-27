import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json


def read_file(file_name):
    return pd.read_csv(file_name, low_memory=False)
    # return next()


def get_memory_stat_by_column(df):
    memory_usage_stat = df.memory_usage(deep=True)
    total_memory_usage = memory_usage_stat.sum()
    print(f"file in memory size = {total_memory_usage // 1024:10} КБ")
    column_stat = list()
    for key in df.dtypes.keys():
        column_stat.append(
            {
                "column_name": key,
                "memory_abs": memory_usage_stat[key] // 1024,
                "memory_per": round(
                    memory_usage_stat[key] / total_memory_usage * 100, 4
                ),
                "dtype": df.dtypes[key],
            }
        )
    column_stat.sort(key=lambda x: x["memory_abs"], reverse=True)
    for column in column_stat:
        print(
            f"{column['column_name']:30}: {column['memory_abs']:10} КБ: {column['memory_per']:10}% : {column['dtype']}"
        )


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:  # предположим, что если это не датафрейм, то серия
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024**2  # преобразуем байты в мегабайты
    return "{:03.2f} MB".format(usage_mb)


def opt_obj(df):
    converted_obj = pd.DataFrame()
    dataset_obj = df.select_dtypes(include=["object"]).copy()

    for col in dataset_obj.columns:
        num_unique_values = len(dataset_obj[col].unique())
        num_total_values = len(dataset_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:, col] = dataset_obj[col].astype("category")
        else:
            converted_obj.loc[:, col] = dataset_obj[col]

    print(mem_usage(dataset_obj))
    print(mem_usage(converted_obj))
    return converted_obj


def opt_int(df):
    dataset_int = df.select_dtypes(include=["int"])
    """
    downcast:
            - 'integer' or 'signed': smallest signed int dtype (min.: np.int8)
            - 'unsigned': smallest unsigned int dtype (min.: np.uint8)
            - 'float': smallest float dtype (min.: np.float32)
    """
    converted_int = dataset_int.apply(pd.to_numeric, downcast="unsigned")
    print(mem_usage(dataset_int))
    print(mem_usage(converted_int))
    #
    compare_ints = pd.concat([dataset_int.dtypes, converted_int.dtypes], axis=1)
    compare_ints.columns = ["before", "after"]
    compare_ints.apply(pd.Series.value_counts)
    print(compare_ints)

    return converted_int


def opt_float(df):
    # # =======================================================================
    # # выполняем понижающее преобразование
    # # для столбцов типа float
    dataset_float = df.select_dtypes(include=["float"])
    converted_float = dataset_float.apply(pd.to_numeric, downcast="float")

    print(mem_usage(dataset_float))
    print(mem_usage(converted_float))

    compare_floats = pd.concat([dataset_float.dtypes, converted_float.dtypes], axis=1)
    compare_floats.columns = ["before", "after"]
    compare_floats.apply(pd.Series.value_counts)
    print(compare_floats)

    return converted_float


# steps 1-3
file_name = "air_quality_new_2.csv"
dataset = read_file(file_name)
get_memory_stat_by_column(dataset)

# steps 4-6
optimized_dataset = dataset.copy()

converted_obj = opt_obj(dataset)
converted_int = opt_int(dataset)
converted_float = opt_float(dataset)
#
optimized_dataset[converted_obj.columns] = converted_obj
optimized_dataset[converted_int.columns] = converted_int
optimized_dataset[converted_float.columns] = converted_float

# 7
get_memory_stat_by_column(dataset)
print(mem_usage(dataset))
print(mem_usage(optimized_dataset))
optimized_dataset.info(memory_usage="deep")

# 8
# отобрать свои 10 колонок
need_column = dict()
column_names = [
    "date",
    "county",
    "aqi",
    "status",
    "co",
    "o3",
    "o3_8hr",
    "pm2.5",
    "no2",
    "nox",
]
opt_dtypes = optimized_dataset.dtypes
for key in dataset.columns:
    need_column[key] = opt_dtypes[key]
    print(f"{key}:{opt_dtypes[key]}")

with open("dtypes_2.json", mode="w") as file:
    dtype_json = need_column.copy()
    for key in dtype_json.keys():
        dtype_json[key] = str(dtype_json[key])

    json.dump(dtype_json, file)

# 9
read_and_optimized = pd.read_csv(
    file_name, usecols=lambda x: x in column_names, dtype=need_column
)

avg_aqi = (
    read_and_optimized.groupby("county", observed=False)["aqi"].mean().sort_values()
)

plt.figure(figsize=(10, 6))
avg_aqi.plot(kind="bar", color="skyblue")
plt.title("Средний AQI по районам")
plt.xlabel("Район")
plt.ylabel("Средний AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

status_counts = read_and_optimized["status"].value_counts()

plt.figure(figsize=(30, 30))
status_counts.plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["green", "yellow", "red", "orange"],
    startangle=140,
)
plt.title("Распределение статуса AQI")
plt.ylabel("")
plt.tight_layout()
plt.show()

read_and_optimized["date"] = pd.to_datetime(
    read_and_optimized["date"], format="mixed", errors="coerce"
)

read_and_optimized["date"] = pd.to_datetime(read_and_optimized["date"])

taichung_data = read_and_optimized[
    read_and_optimized["county"] == "Taichung City"
].sort_values("date")

plt.figure(figsize=(20, 6))
plt.plot(taichung_data["date"], taichung_data["aqi"], marker="o", linestyle="-")
plt.title("Изменение AQI во времени для Taichung City")
plt.xlabel("Дата")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

read_and_optimized["pm2.5"] = pd.to_numeric(
    read_and_optimized["pm2.5"], errors="coerce"
)

plt.figure(figsize=(12, 6))

sns.histplot(read_and_optimized["pm2.5"], bins=30, kde=True, color="blue")

x_ticks = np.arange(0, read_and_optimized["pm2.5"].max() + 5, 5)
plt.xticks(x_ticks, rotation=45)

plt.xlabel("PM2.5 Concentration")
plt.ylabel("Frequency")
plt.title("Distribution of PM2.5 Particles")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x="status", y="aqi", data=read_and_optimized, palette="Set2")
plt.title("Распределение AQI по статусам")
plt.xlabel("Статус AQI")
plt.ylabel("AQI")
plt.tight_layout()
plt.show()
