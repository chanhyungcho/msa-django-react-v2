from django.db import models

from blog.b_users.models import BUser


class Post(models.Model):
    use_in_migrations = True
    post_id = models.AutoField(primary_key=True)
    title = models.CharField(max_length=100)
    content = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


    blog_user = models.ForeignKey(BUser, on_delete=models.CASCADE)



    class Meta:
        db_table = "blog_posts"

    def __str__(self):
        return f'{self.pk}{self.title}{self.content}{self.created_at}{self.updated_at}'
