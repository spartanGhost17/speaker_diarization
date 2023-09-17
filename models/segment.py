class AudioSegment:
    def __init__(self, id, start, end, text, temperature, avg_logprob, compression_ratio, no_speech_prob):
        self.start = start
        self.end = end
        self.text = text
        self.temperature = temperature
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob
        self.id = id

    def get_id(self):
        return self.id

    def get_start(self):
        return self.start

    def get_end(self):
        return self.end
    
    def get_text(self):
        return f"{self.text}"
    
    def get_temperature(self):
        return self.temperature
    
    def get_avg_logprob(self):
        return self.avg_logprob
    
    def get_compression_ratio(self):
        return self.compression_ratio
    
    def __str__(self):
        return f"id: {self.id}, start: {self.start}, end: {self.end}, text: {self.text}, temperature: {self.temperature}, avg_logprob: {self.avg_logprob}, compression_ratio: {self.compression_ratio}"