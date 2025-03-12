import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)


GRAPH_FILE_PATH = "data/transport_network.graphml"
DATA_FILE_PATH = "data/connection_graph.csv"
