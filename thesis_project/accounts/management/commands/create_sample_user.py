from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

User = get_user_model()

class Command(BaseCommand):
    help = 'Create sample users for testing'

    def handle(self, *args, **options):
        if not User.objects.filter(email='admin@wmsu.edu.ph').exists():
            admin_user = User.objects.create_superuser(
                username='admin',
                email='admin@wmsu.edu.ph',
                password='password',
                first_name='Admin',
                last_name='User',
                role='admin'
            )
            self.stdout.write(self.style.SUCCESS(f'Successfully created admin user: {admin_user.email}'))
        else:
            self.stdout.write(self.style.WARNING('Admin user already exists'))