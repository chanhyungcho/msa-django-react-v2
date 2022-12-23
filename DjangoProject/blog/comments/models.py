from django.db import models

from blog.b_users.models import BUser
from blog.posts.models import Post


class Comment(models.Model):
    use_in_migrations = True
    comment_id = models.AutoField(primary_key=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    parent_id = models.TextField(null=True)

    blog_user = models.ForeignKey(BUser, on_delete=models.CASCADE)
    blog_post = models.ForeignKey(Post, on_delete=models.CASCADE)


    class Meta:
        db_table = "blog_comments"

    def __str__(self):
        return f'{self.pk}{self.content}{self.created_at}{self.updated_at}{self.parent_id}'
