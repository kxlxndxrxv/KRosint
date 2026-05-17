import streamlit as st
import os
import glob
from PIL import Image
import sys
import io
from contextlib import redirect_stdout

import scraper
import analyzer

st.set_page_config(page_title="KRosint — Анализ никнеймов", layout="wide", page_icon="📊")

st.title("KRosint — Анализ эволюции никнеймов (2000-2026)")
st.markdown("Это приложение автоматически собирает профили с IT-ресурсов (Slashdot, SourceForge, HackerNews, GitHub) и анализирует структурные изменения никнеймов за 26 лет.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Управление")
    
    if st.button("📡 Собрать данные", width="stretch", type="primary"):
        st.info("Начат сбор данных. Пожалуйста, не закрывайте страницу...")
        log_container = st.empty()
        
        # Перехватываем stdout чтобы показать логи
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                scraper.main()
                st.success("✅ Сбор данных успешно завершен!")
            except Exception as e:
                st.error(f"❌ Ошибка при сборе данных: {e}")
        
        # Показываем логи после завершения
        with st.expander("Логи сбора данных", expanded=False):
            st.code(f.getvalue(), language="text")

    if st.button("📊 Провести анализ", width="stretch", type="primary"):
        st.info("Начат анализ данных. Генерируются графики...")
        f = io.StringIO()
        with redirect_stdout(f):
            try:
                analyzer.main()
                st.success("✅ Анализ успешно завершен!")
            except Exception as e:
                st.error(f"❌ Ошибка при анализе: {e}")
                
        with st.expander("Логи анализа", expanded=False):
            st.code(f.getvalue(), language="text")
                
    st.markdown("---")
    st.subheader("Сводный текстовый отчёт")
    if os.path.exists("output/report.txt"):
        with open("output/report.txt", "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.info("Отчет появится здесь после запуска анализа.")

with col2:
    st.header("Галерея графиков")
    
    if not os.path.exists("output"):
        os.makedirs("output")
        
    images = sorted(glob.glob("output/*.png"))
    
    if not images:
        st.info("Здесь появятся графики после завершения анализа.")
    else:
        # Сделаем табы (вкладки) для каждого графика, чтобы не прокручивать вечность
        tabs = st.tabs([os.path.basename(img).replace(".png", "") for img in images])
        
        for tab, img_path in zip(tabs, images):
            with tab:
                st.image(img_path, width="stretch")
