

from django.db import models

class MUser(models.Model):
    use_in_migrations = True
    m_userid = models.AutoField(primary_key=True)
    email = models.TextField()
    nickname = models.TextField()
    password = models.TextField()
    age = models.TextField()

    class Meta:
        db_table = "m_users"

    def __str__(self):
        return f'{self.pk}{self.email}{self.nickname}{self.password}{self.age}'











