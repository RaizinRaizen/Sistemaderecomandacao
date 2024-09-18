import numpy as np
from sklearn.cluster import KMeans


filmes_assistidos = np.array([
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],  # usuário 1
    [1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1],  # usuário 2
    [0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],  # usuário 3
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],  # usuário 4
    [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1],  # usuário 5
    [0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0],  # usuário 6
    [1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0],  # usuário 7
    [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1],  # usuário 8
    [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1],  # usuário 9
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],  # usuário 10
    [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0],  # usuário 11
    [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],  # usuário 12
    [1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0],  # usuário 13
    [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],  # usuário 14
    [1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0],  # usuário 15
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1],  # usuário 16
    [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0],  # usuário 17
    [0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0],  # usuário 18
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1],  # usuário 19
    [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1],  # usuário 20
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0],  # usuário 21
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1],  # usuário 22
    [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0],  # usuário 23
    [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],  # usuário 24
    [1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1],  # usuário 25
])

num_clusters = 6 

kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
kmeans.fit(filmes_assistidos)

grupos_indice = kmeans.predict(filmes_assistidos)

print("Usuário pertence ao seguinte grupo:")
for i, cluster in enumerate(grupos_indice):
    print(f"Usuário {i + 1} pertence ao grupo {cluster + 1}")

print("\nFilmes assistidos:")
for i in range(len(filmes_assistidos)):
    assistidos = np.where(filmes_assistidos[i] == 1)[0] + 1
    print(f"Usuário {i + 1} assistiu aos filmes: {assistidos}")

def recomendar_filmes(filmes, filmes_assistidos, grupos_indice):
    filmes = np.array(filmes)
    grupo_usuario = kmeans.predict([filmes])[0]

    usuarios_no_mesmo_grupo = [i for i in range(len(grupos_indice)) if grupos_indice[i] == grupo_usuario]

    filmes_recomendados = set()
    for usuario in usuarios_no_mesmo_grupo:
        filmes_assistidos_usuario = np.where(filmes_assistidos[usuario] == 1)[0]
        filmes_recomendados.update(filmes_assistidos_usuario)

    filmes_recomendados = filmes_recomendados - set(np.where(filmes == 1)[0])

    return sorted([filme + 1 for filme in filmes_recomendados])


filmes_novos = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
recomendacoes = recomendar_filmes(filmes_novos, filmes_assistidos, grupos_indice)
print(f"Filmes recomendados: {recomendacoes}")

filmes_assistidos_usuario = [1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
filmes_recomendados = recomendar_filmes(filmes_assistidos_usuario, filmes_assistidos, grupos_indice)
print(f"\nFilmes recomendados: {filmes_recomendados}")

