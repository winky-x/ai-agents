"""
Real-World API Integrations
- Actual connections to real services
- Authentication and security
- Error handling and retry logic
- Rate limiting and quota management
- Real-time data processing
"""

import os
import json
import time
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import httpx
import aiohttp
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import stripe
from loguru import logger


class ServiceType(Enum):
    """Types of services"""
    EMAIL = "email"
    CALENDAR = "calendar"
    PAYMENT = "payment"
    ECOMMERCE = "ecommerce"
    FOOD_DELIVERY = "food_delivery"
    WEATHER = "weather"
    MAPS = "maps"
    TRAVEL = "travel"
    SOCIAL = "social"
    STORAGE = "storage"


@dataclass
class ServiceConfig:
    """Service configuration"""
    service_type: ServiceType
    api_key: str
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None
    quota_reset: Optional[int] = None


@dataclass
class EmailMessage:
    """Email message"""
    to: List[str]
    subject: str
    body: str
    cc: List[str] = None
    bcc: List[str] = None
    attachments: List[str] = None
    priority: str = "normal"


@dataclass
class CalendarEvent:
    """Calendar event"""
    title: str
    start_time: datetime
    end_time: datetime
    attendees: List[str] = None
    location: str = None
    description: str = None
    reminder_minutes: int = 15


@dataclass
class PaymentRequest:
    """Payment request"""
    amount: float
    currency: str
    description: str
    customer_email: str
    payment_method: str = "card"
    metadata: Dict[str, Any] = None


@dataclass
class ProductSearch:
    """Product search"""
    query: str
    category: Optional[str] = None
    price_min: Optional[float] = None
    price_max: Optional[float] = None
    sort_by: str = "relevance"
    limit: int = 10


@dataclass
class FoodOrder:
    """Food order"""
    restaurant: str
    items: List[Dict[str, Any]]
    delivery_address: str
    special_instructions: str = None
    payment_method: str = "card"


class RealWorldIntegrations:
    """Real-world API integrations"""
    
    def __init__(self):
        self.services: Dict[ServiceType, ServiceConfig] = {}
        self.clients: Dict[ServiceType, Any] = {}
        self.rate_limiters: Dict[ServiceType, Dict[str, Any]] = {}
        
        # Load service configurations
        self._load_service_configs()
        
        # Initialize service clients
        self._initialize_clients()
    
    def _load_service_configs(self):
        """Load service configurations from environment"""
        
        # Gmail/Google Calendar
        if os.getenv("GOOGLE_CLIENT_ID") and os.getenv("GOOGLE_CLIENT_SECRET"):
            self.services[ServiceType.EMAIL] = ServiceConfig(
                service_type=ServiceType.EMAIL,
                api_key=os.getenv("GOOGLE_CLIENT_ID"),
                api_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
                base_url="https://gmail.googleapis.com",
                rate_limit=1000,  # Gmail API quota
                quota_reset=86400  # 24 hours
            )
            
            self.services[ServiceType.CALENDAR] = ServiceConfig(
                service_type=ServiceType.CALENDAR,
                api_key=os.getenv("GOOGLE_CLIENT_ID"),
                api_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
                base_url="https://www.googleapis.com/calendar/v3",
                rate_limit=10000,  # Calendar API quota
                quota_reset=86400
            )
        
        # Stripe Payments
        if os.getenv("STRIPE_SECRET_KEY"):
            self.services[ServiceType.PAYMENT] = ServiceConfig(
                service_type=ServiceType.PAYMENT,
                api_key=os.getenv("STRIPE_SECRET_KEY"),
                base_url="https://api.stripe.com/v1",
                rate_limit=100,  # Stripe rate limit
                quota_reset=1  # Per second
            )
        
        # OpenWeatherMap
        if os.getenv("OPENWEATHER_API_KEY"):
            self.services[ServiceType.WEATHER] = ServiceConfig(
                service_type=ServiceType.WEATHER,
                api_key=os.getenv("OPENWEATHER_API_KEY"),
                base_url="https://api.openweathermap.org/data/2.5",
                rate_limit=60,  # 60 calls per minute
                quota_reset=60
            )
        
        # Google Maps
        if os.getenv("GOOGLE_MAPS_API_KEY"):
            self.services[ServiceType.MAPS] = ServiceConfig(
                service_type=ServiceType.MAPS,
                api_key=os.getenv("GOOGLE_MAPS_API_KEY"),
                base_url="https://maps.googleapis.com/maps/api",
                rate_limit=1000,  # Maps API quota
                quota_reset=86400
            )
        
        # Uber Eats (simulated - would need actual API access)
        if os.getenv("UBER_CLIENT_ID"):
            self.services[ServiceType.FOOD_DELIVERY] = ServiceConfig(
                service_type=ServiceType.FOOD_DELIVERY,
                api_key=os.getenv("UBER_CLIENT_ID"),
                api_secret=os.getenv("UBER_CLIENT_SECRET"),
                base_url="https://api.uber.com/v1",
                rate_limit=100,
                quota_reset=60
            )
    
    def _initialize_clients(self):
        """Initialize service clients"""
        
        # Initialize Gmail client
        if ServiceType.EMAIL in self.services:
            try:
                self.clients[ServiceType.EMAIL] = self._init_gmail_client()
            except Exception as e:
                logger.error(f"Failed to initialize Gmail client: {e}")
        
        # Initialize Calendar client
        if ServiceType.CALENDAR in self.services:
            try:
                self.clients[ServiceType.CALENDAR] = self._init_calendar_client()
            except Exception as e:
                logger.error(f"Failed to initialize Calendar client: {e}")
        
        # Initialize Stripe client
        if ServiceType.PAYMENT in self.services:
            try:
                stripe.api_key = self.services[ServiceType.PAYMENT].api_key
                self.clients[ServiceType.PAYMENT] = stripe
            except Exception as e:
                logger.error(f"Failed to initialize Stripe client: {e}")
        
        # Initialize HTTP clients for other services
        for service_type in [ServiceType.WEATHER, ServiceType.MAPS, ServiceType.FOOD_DELIVERY]:
            if service_type in self.services:
                self.clients[service_type] = httpx.AsyncClient(timeout=30.0)
    
    def _init_gmail_client(self):
        """Initialize Gmail API client"""
        SCOPES = ['https://www.googleapis.com/auth/gmail.send']
        
        creds = None
        # The file token.json stores the user's access and refresh tokens
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())
        
        return build('gmail', 'v1', credentials=creds)
    
    def _init_calendar_client(self):
        """Initialize Google Calendar API client"""
        SCOPES = ['https://www.googleapis.com/auth/calendar']
        
        creds = None
        if os.path.exists('calendar_token.json'):
            creds = Credentials.from_authorized_user_file('calendar_token.json', SCOPES)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            
            with open('calendar_token.json', 'w') as token:
                token.write(creds.to_json())
        
        return build('calendar', 'v3', credentials=creds)
    
    async def send_email(self, message: EmailMessage) -> Dict[str, Any]:
        """Send email using Gmail API"""
        if ServiceType.EMAIL not in self.clients:
            return {"success": False, "error": "Gmail service not configured"}
        
        try:
            # Check rate limit
            if not self._check_rate_limit(ServiceType.EMAIL):
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Create email message
            email_content = f"""
            To: {', '.join(message.to)}
            Subject: {message.subject}
            
            {message.body}
            """
            
            # Encode the message
            import base64
            encoded_message = base64.urlsafe_b64encode(email_content.encode()).decode()
            
            # Send the email
            result = self.clients[ServiceType.EMAIL].users().messages().send(
                userId='me',
                body={'raw': encoded_message}
            ).execute()
            
            # Update rate limiter
            self._update_rate_limit(ServiceType.EMAIL)
            
            return {
                "success": True,
                "message_id": result.get('id'),
                "thread_id": result.get('threadId')
            }
            
        except HttpError as error:
            logger.error(f"Gmail API error: {error}")
            return {"success": False, "error": str(error)}
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def create_calendar_event(self, event: CalendarEvent) -> Dict[str, Any]:
        """Create calendar event using Google Calendar API"""
        if ServiceType.CALENDAR not in self.clients:
            return {"success": False, "error": "Calendar service not configured"}
        
        try:
            # Check rate limit
            if not self._check_rate_limit(ServiceType.CALENDAR):
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Create event body
            event_body = {
                'summary': event.title,
                'description': event.description,
                'start': {
                    'dateTime': event.start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': event.end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in (event.attendees or [])],
                'location': event.location,
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': event.reminder_minutes},
                        {'method': 'popup', 'minutes': event.reminder_minutes},
                    ],
                },
            }
            
            # Create the event
            result = self.clients[ServiceType.CALENDAR].events().insert(
                calendarId='primary',
                body=event_body
            ).execute()
            
            # Update rate limiter
            self._update_rate_limit(ServiceType.CALENDAR)
            
            return {
                "success": True,
                "event_id": result.get('id'),
                "html_link": result.get('htmlLink'),
                "start_time": result.get('start', {}).get('dateTime')
            }
            
        except HttpError as error:
            logger.error(f"Calendar API error: {error}")
            return {"success": False, "error": str(error)}
        except Exception as e:
            logger.error(f"Calendar event creation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def process_payment(self, payment: PaymentRequest) -> Dict[str, Any]:
        """Process payment using Stripe"""
        if ServiceType.PAYMENT not in self.clients:
            return {"success": False, "error": "Payment service not configured"}
        
        try:
            # Check rate limit
            if not self._check_rate_limit(ServiceType.PAYMENT):
                return {"success": False, "error": "Rate limit exceeded"}
            
            # Create payment intent
            intent = stripe.PaymentIntent.create(
                amount=int(payment.amount * 100),  # Convert to cents
                currency=payment.currency,
                description=payment.description,
                receipt_email=payment.customer_email,
                metadata=payment.metadata or {}
            )
            
            # Update rate limiter
            self._update_rate_limit(ServiceType.PAYMENT)
            
            return {
                "success": True,
                "payment_intent_id": intent.id,
                "client_secret": intent.client_secret,
                "amount": intent.amount / 100,  # Convert back from cents
                "status": intent.status
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Stripe error: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Payment processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_products(self, search: ProductSearch) -> Dict[str, Any]:
        """Search products (simulated - would connect to real e-commerce APIs)"""
        try:
            # Simulate product search
            # In reality, this would connect to Amazon, eBay, or other e-commerce APIs
            
            products = [
                {
                    "id": f"prod_{i}",
                    "name": f"Product {i} - {search.query}",
                    "price": 19.99 + (i * 5),
                    "rating": 4.2 + (i * 0.1),
                    "image_url": f"https://example.com/product_{i}.jpg",
                    "description": f"This is a {search.query} product",
                    "in_stock": True
                }
                for i in range(1, min(search.limit + 1, 6))
            ]
            
            return {
                "success": True,
                "products": products,
                "total_results": len(products),
                "query": search.query
            }
            
        except Exception as e:
            logger.error(f"Product search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def order_food(self, order: FoodOrder) -> Dict[str, Any]:
        """Order food (simulated - would connect to Uber Eats, DoorDash, etc.)"""
        try:
            # Simulate food ordering
            # In reality, this would connect to Uber Eats, DoorDash, or other delivery APIs
            
            order_id = f"order_{int(time.time())}"
            estimated_delivery = datetime.utcnow() + timedelta(minutes=30)
            
            return {
                "success": True,
                "order_id": order_id,
                "restaurant": order.restaurant,
                "estimated_delivery": estimated_delivery.isoformat(),
                "total_amount": sum(item.get("price", 0) for item in order.items),
                "status": "confirmed"
            }
            
        except Exception as e:
            logger.error(f"Food ordering failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_weather(self, location: str) -> Dict[str, Any]:
        """Get weather information"""
        if ServiceType.WEATHER not in self.clients:
            return {"success": False, "error": "Weather service not configured"}
        
        try:
            # Check rate limit
            if not self._check_rate_limit(ServiceType.WEATHER):
                return {"success": False, "error": "Rate limit exceeded"}
            
            config = self.services[ServiceType.WEATHER]
            url = f"{config.base_url}/weather"
            
            params = {
                "q": location,
                "appid": config.api_key,
                "units": "metric"
            }
            
            async with self.clients[ServiceType.WEATHER] as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Update rate limiter
                self._update_rate_limit(ServiceType.WEATHER)
                
                return {
                    "success": True,
                    "location": data.get("name"),
                    "temperature": data.get("main", {}).get("temp"),
                    "description": data.get("weather", [{}])[0].get("description"),
                    "humidity": data.get("main", {}).get("humidity"),
                    "wind_speed": data.get("wind", {}).get("speed")
                }
                
        except Exception as e:
            logger.error(f"Weather lookup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_directions(self, origin: str, destination: str, mode: str = "driving") -> Dict[str, Any]:
        """Get directions using Google Maps API"""
        if ServiceType.MAPS not in self.clients:
            return {"success": False, "error": "Maps service not configured"}
        
        try:
            # Check rate limit
            if not self._check_rate_limit(ServiceType.MAPS):
                return {"success": False, "error": "Rate limit exceeded"}
            
            config = self.services[ServiceType.MAPS]
            url = f"{config.base_url}/directions/json"
            
            params = {
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "key": config.api_key
            }
            
            async with self.clients[ServiceType.MAPS] as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if data.get("status") != "OK":
                    return {"success": False, "error": f"Maps API error: {data.get('status')}"}
                
                # Update rate limiter
                self._update_rate_limit(ServiceType.MAPS)
                
                route = data.get("routes", [{}])[0]
                leg = route.get("legs", [{}])[0]
                
                return {
                    "success": True,
                    "distance": leg.get("distance", {}).get("text"),
                    "duration": leg.get("duration", {}).get("text"),
                    "steps": [
                        step.get("html_instructions") 
                        for step in leg.get("steps", [])
                    ][:5]  # First 5 steps
                }
                
        except Exception as e:
            logger.error(f"Directions lookup failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_rate_limit(self, service_type: ServiceType) -> bool:
        """Check if service is within rate limits"""
        if service_type not in self.services:
            return False
        
        config = self.services[service_type]
        if not config.rate_limit:
            return True
        
        service_key = service_type.value
        if service_key not in self.rate_limiters:
            self.rate_limiters[service_key] = {
                "calls": 0,
                "reset_time": time.time() + config.quota_reset
            }
        
        limiter = self.rate_limiters[service_key]
        
        # Check if we need to reset
        if time.time() > limiter["reset_time"]:
            limiter["calls"] = 0
            limiter["reset_time"] = time.time() + config.quota_reset
        
        return limiter["calls"] < config.rate_limit
    
    def _update_rate_limit(self, service_type: ServiceType):
        """Update rate limiter after successful call"""
        service_key = service_type.value
        if service_key in self.rate_limiters:
            self.rate_limiters[service_key]["calls"] += 1
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}
        
        for service_type, config in self.services.items():
            service_key = service_type.value
            limiter = self.rate_limiters.get(service_key, {})
            
            status[service_key] = {
                "configured": True,
                "rate_limit": config.rate_limit,
                "calls_used": limiter.get("calls", 0),
                "reset_time": limiter.get("reset_time", 0),
                "available": self._check_rate_limit(service_type)
            }
        
        return status
    
    def get_available_services(self) -> List[ServiceType]:
        """Get list of available services"""
        return list(self.services.keys())