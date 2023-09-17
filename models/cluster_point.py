class ClusterPoint():
    def __init__(self):
        self.filename = None
        self.start = None
        self.end = None
        self.counter = None
        self.cluster_label = None

    def add_file_path(self, filename):
        self.filename = filename

    def add_start(self, start):
        self.start = start
    
    def add_end(self, end):
        self.end = end

    def add_counter(self, counter):
        self.counter = counter
    
    def add_cluster_label(self, cluster_label):
        self.cluster_label = cluster_label

    def get_start(self):
        return self.start
    
    def get_end(self):
        return self.end
    
    def get_counter(self):
        return self.counter

    def get_file_path(self):
        return self.filename

    def get_cluster_label(self):
        return self.cluster_label
    
    def __str__(self):
        return f"({self.filename}, ({self.start}, {self.end}, {self.counter}, {self.cluster_label})"