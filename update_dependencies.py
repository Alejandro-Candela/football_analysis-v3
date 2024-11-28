import os
import subprocess


def update_requirements():
    """Actualiza el archivo requirements.txt con los paquetes instalados actualmente usando pip."""
    print("Actualizando requirements.txt...")
    with open("requirements.txt", "w") as req_file:
        subprocess.run(["pip", "freeze"], stdout=req_file)
    print("Archivo requirements.txt actualizado.")


def update_environment_yml():
    """Actualiza el archivo environment.yml con las dependencias del entorno conda."""
    print("Actualizando environment.yml...")
    with open("environment.yml", "w") as env_file:
        subprocess.run(["conda", "env", "export", "--no-builds"], stdout=env_file)
    print("Archivo environment.yml actualizado.")


def update_env_windows_yml():
    """Actualiza el archivo env_windows.yml con las dependencias principales usando --from-history."""
    print("Actualizando env_windows.yml...")
    with open("env_windows.yml", "w") as env_file:
        subprocess.run(["conda", "env", "export", "--from-history"], stdout=env_file)
    print("Archivo env_windows.yml actualizado.")


def main():
    """Ejecuta las actualizaciones de archivos de dependencias."""
    # Crear los archivos si no existen
    for file_name in ["requirements.txt", "environment.yml", "env_windows.yml"]:
        if not os.path.exists(file_name):
            print(f"Creando un archivo {file_name}...")
            open(file_name, "w").close()

    # Actualizar los archivos
    update_requirements()
    update_environment_yml()
    update_env_windows_yml()

    print("\n¡Todos los archivos de dependencias han sido actualizados correctamente!")


if __name__ == "__main__":
    main()
