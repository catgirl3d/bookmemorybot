import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# Проверка наличия API ключа
if not os.environ.get("OPENAI_API_KEY"):
    print("ВНИМАНИЕ: Переменная окружения OPENAI_API_KEY не найдена!")
    print("Вы можете установить её любым из следующих способов:")
    print("1. Создать файл .env в корневой директории проекта с содержимым:")
    print("   OPENAI_API_KEY=ваш_ключ_api")
    print("2. Установить переменную окружения перед запуском программы:")
    print("   В Linux/Mac: export OPENAI_API_KEY=ваш_ключ_api")
    print("   В Windows (CMD): set OPENAI_API_KEY=ваш_ключ_api")
    print("   В Windows (PowerShell): $env:OPENAI_API_KEY='ваш_ключ_api'")
    api_key = input("\nВведите ваш OpenAI API ключ сейчас (или оставьте пустым для выхода): ")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("API ключ не предоставлен. Выход.")
        exit(1)

# Импортируем остальные модули после установки API ключа
import asyncio
import main

if __name__ == "__main__":
    try:
        asyncio.run(main.main())
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
    except Exception as e:
        print(f"Произошла критическая ошибка: {e}")
        print("Проверьте файл логов app.log для получения дополнительной информации.")
