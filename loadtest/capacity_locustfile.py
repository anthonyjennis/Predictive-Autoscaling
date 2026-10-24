from locust import HttpUser, task, constant_throughput

class User(HttpUser):
    wait_time = constant_throughput(1)   # each user = 1 request/second

    @task
    def hit(self):
        self.client.get("/")