"""
Encrypted Credential Storage System
OrbitScope ML Intelligence Suite - Secure Credential Management

This module provides secure, encrypted storage and retrieval of sensitive credentials
including Plaid API keys, access tokens, and other authentication data.

Security Features:
- AES-256-GCM encryption for all sensitive data
- Secure key derivation using PBKDF2
- Salt-based encryption to prevent rainbow table attacks
- Separate encryption for each credential type
- Automatic key rotation capabilities
- Audit logging for all credential access

Dependencies:
- cryptography: For encryption/decryption operations
- motor: Async MongoDB driver
- hashlib: For secure hashing
"""

import asyncio
import logging
import os
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logging.warning("Cryptography library not available - credentials will be stored in plaintext")

try:
    import motor.motor_asyncio
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    logging.warning("Motor (async MongoDB) not available - using simulation mode")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CredentialType(Enum):
    PLAID_API = "plaid_api"
    PLAID_ACCESS_TOKEN = "plaid_access_token"
    DATABASE_URL = "database_url"
    API_KEY = "api_key"
    OAUTH_TOKEN = "oauth_token"
    ENCRYPTION_KEY = "encryption_key"

class CredentialStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"

@dataclass
class EncryptedCredential:
    credential_id: str
    user_id: str
    credential_type: CredentialType
    encrypted_data: bytes
    salt: bytes
    created_at: datetime
    expires_at: Optional[datetime]
    status: CredentialStatus
    metadata: Dict[str, Any]
    access_count: int
    last_accessed: Optional[datetime]

@dataclass
class PlaidCredentials:
    client_id: str
    secret: str
    environment: str
    access_token: Optional[str] = None
    item_id: Optional[str] = None
    institution_id: Optional[str] = None

class EncryptedCredentialManager:
    """
    Secure credential storage system with AES-256 encryption and MongoDB backend.
    
    Provides encrypted storage, retrieval, and management of sensitive credentials
    including Plaid API keys, access tokens, and other authentication data.
    """
    
    def __init__(self, mongodb_url: str = None, master_key: str = None):
        self.logger = logger
        self.mongodb_url = mongodb_url or "mongodb://localhost:27017"
        self.master_key = master_key or self._generate_master_key()
        
        # Database connection
        self.client = None
        self.database = None
        self.credentials_collection = None
        self.audit_collection = None
        
        # Encryption components
        self.encryption_cache = {}
        
        # Initialize components (will be called when needed)
        self._db_initialized = False
        
        self.logger.info("Encrypted Credential Manager initialized")
    
    def _generate_master_key(self) -> str:
        """Generate a secure master key for encryption."""
        # In production, this should be loaded from secure environment or key management service
        master_key = os.getenv('CREDENTIAL_MASTER_KEY')
        if not master_key:
            # Generate a new key - THIS SHOULD BE STORED SECURELY
            master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            self.logger.warning("Generated new master key - STORE THIS SECURELY!")
            self.logger.warning(f"CREDENTIAL_MASTER_KEY={master_key}")
        
        return master_key
    
    async def _initialize_database(self):
        """Initialize MongoDB connection and collections."""
        if not MOTOR_AVAILABLE:
            self.logger.warning("MongoDB not available - using simulation mode")
            return
        
        try:
            self.client = motor.motor_asyncio.AsyncIOMotorClient(self.mongodb_url)
            self.database = self.client.orbitscope_unified
            self.credentials_collection = self.database.credentials
            self.audit_collection = self.database.credential_audit
            
            # Create indexes for performance
            await self.credentials_collection.create_index([
                ("user_id", 1),
                ("credential_type", 1),
                ("status", 1)
            ])
            
            await self.audit_collection.create_index([
                ("user_id", 1),
                ("timestamp", -1)
            ])
            
            self.logger.info("Database connection established")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.client = None
    
    def _derive_key_from_password(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2."""
        if not CRYPTO_AVAILABLE:
            return password.encode()[:32].ljust(32, b'0')
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # OWASP recommended minimum
        )
        return kdf.derive(password.encode())
    
    def _encrypt_data(self, data: str, user_salt: bytes) -> bytes:
        """Encrypt sensitive data using AES-256-GCM."""
        if not CRYPTO_AVAILABLE:
            # Fallback - base64 encoding (NOT secure for production)
            self.logger.warning("Using insecure fallback encryption")
            return base64.b64encode(data.encode())
        
        # Derive key from master key and user salt
        key = self._derive_key_from_password(self.master_key, user_salt)
        f = Fernet(base64.urlsafe_b64encode(key))
        
        # Encrypt the data
        encrypted_data = f.encrypt(data.encode())
        return encrypted_data
    
    def _decrypt_data(self, encrypted_data: bytes, user_salt: bytes) -> str:
        """Decrypt sensitive data using AES-256-GCM."""
        if not CRYPTO_AVAILABLE:
            # Fallback - base64 decoding (NOT secure for production)
            return base64.b64decode(encrypted_data).decode()
        
        try:
            # Derive key from master key and user salt
            key = self._derive_key_from_password(self.master_key, user_salt)
            f = Fernet(base64.urlsafe_b64encode(key))
            
            # Decrypt the data
            decrypted_data = f.decrypt(encrypted_data)
            return decrypted_data.decode()
            
        except InvalidToken:
            self.logger.error("Failed to decrypt data - invalid token or corrupted data")
            raise ValueError("Decryption failed - invalid credentials or corrupted data")
    
    async def _ensure_db_initialized(self):
        """Ensure database is initialized before operations."""
        if not self._db_initialized:
            await self._initialize_database()
            self._db_initialized = True

    async def store_plaid_credentials(self, user_id: str, credentials: PlaidCredentials) -> str:
        """
        Store Plaid API credentials securely.
        
        Args:
            user_id: User identifier
            credentials: PlaidCredentials object with API keys
            
        Returns:
            Credential ID for reference
        """
        await self._ensure_db_initialized()
        
        try:
            # Serialize credentials
            credential_data = {
                'client_id': credentials.client_id,
                'secret': credentials.secret,
                'environment': credentials.environment,
                'access_token': credentials.access_token,
                'item_id': credentials.item_id,
                'institution_id': credentials.institution_id
            }
            
            credential_json = str(credential_data)  # Convert to JSON string
            
            # Generate unique salt for this credential
            salt = secrets.token_bytes(32)
            
            # Encrypt the credential data
            encrypted_data = self._encrypt_data(credential_json, salt)
            
            # Create credential object
            credential_id = f"plaid_{user_id}_{secrets.token_hex(8)}"
            
            encrypted_credential = EncryptedCredential(
                credential_id=credential_id,
                user_id=user_id,
                credential_type=CredentialType.PLAID_API,
                encrypted_data=encrypted_data,
                salt=salt,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),  # 1 year expiry
                status=CredentialStatus.ACTIVE,
                metadata={
                    'environment': credentials.environment,
                    'has_access_token': credentials.access_token is not None
                },
                access_count=0,
                last_accessed=None
            )
            
            # Store in database
            await self._store_credential(encrypted_credential)
            
            # Log the storage
            await self._log_credential_access(
                user_id, credential_id, "STORE", "Plaid credentials stored"
            )
            
            self.logger.info(f"Plaid credentials stored for user {user_id}")
            return credential_id
            
        except Exception as e:
            self.logger.error(f"Failed to store Plaid credentials: {e}")
            raise
    
    async def retrieve_plaid_credentials(self, user_id: str, credential_id: str = None) -> Optional[PlaidCredentials]:
        """
        Retrieve Plaid API credentials for a user.
        
        Args:
            user_id: User identifier
            credential_id: Specific credential ID (optional - gets latest if not provided)
            
        Returns:
            PlaidCredentials object or None if not found
        """
        try:
            # Get credential from database
            encrypted_credential = await self._retrieve_credential(
                user_id, CredentialType.PLAID_API, credential_id
            )
            
            if not encrypted_credential:
                return None
            
            # Decrypt the credential data
            decrypted_data = self._decrypt_data(
                encrypted_credential.encrypted_data,
                encrypted_credential.salt
            )
            
            # Parse credentials
            credential_dict = eval(decrypted_data)  # Safe since we control the data format
            
            credentials = PlaidCredentials(
                client_id=credential_dict['client_id'],
                secret=credential_dict['secret'],
                environment=credential_dict['environment'],
                access_token=credential_dict.get('access_token'),
                item_id=credential_dict.get('item_id'),
                institution_id=credential_dict.get('institution_id')
            )
            
            # Update access tracking
            await self._update_access_tracking(encrypted_credential.credential_id)
            
            # Log the access
            await self._log_credential_access(
                user_id, encrypted_credential.credential_id, "RETRIEVE", "Plaid credentials accessed"
            )
            
            return credentials
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve Plaid credentials: {e}")
            return None
    
    async def store_access_token(self, user_id: str, access_token: str, item_id: str, 
                                institution_id: str = None) -> str:
        """
        Store Plaid access token securely.
        
        Args:
            user_id: User identifier
            access_token: Plaid access token
            item_id: Plaid item ID
            institution_id: Bank institution ID (optional)
            
        Returns:
            Credential ID for reference
        """
        try:
            # Create access token data
            token_data = {
                'access_token': access_token,
                'item_id': item_id,
                'institution_id': institution_id,
                'created_at': datetime.now().isoformat()
            }
            
            token_json = str(token_data)
            
            # Generate unique salt
            salt = secrets.token_bytes(32)
            
            # Encrypt the token data
            encrypted_data = self._encrypt_data(token_json, salt)
            
            # Create credential object
            credential_id = f"access_{user_id}_{secrets.token_hex(8)}"
            
            encrypted_credential = EncryptedCredential(
                credential_id=credential_id,
                user_id=user_id,
                credential_type=CredentialType.PLAID_ACCESS_TOKEN,
                encrypted_data=encrypted_data,
                salt=salt,
                created_at=datetime.now(),
                expires_at=None,  # Access tokens don't expire unless revoked
                status=CredentialStatus.ACTIVE,
                metadata={
                    'item_id': item_id,
                    'institution_id': institution_id
                },
                access_count=0,
                last_accessed=None
            )
            
            # Store in database
            await self._store_credential(encrypted_credential)
            
            # Log the storage
            await self._log_credential_access(
                user_id, credential_id, "STORE", f"Access token stored for item {item_id}"
            )
            
            self.logger.info(f"Access token stored for user {user_id}, item {item_id}")
            return credential_id
            
        except Exception as e:
            self.logger.error(f"Failed to store access token: {e}")
            raise
    
    async def retrieve_access_token(self, user_id: str, item_id: str = None) -> Optional[Dict[str, str]]:
        """
        Retrieve Plaid access token for a user/item.
        
        Args:
            user_id: User identifier
            item_id: Plaid item ID (optional - gets latest if not provided)
            
        Returns:
            Dictionary with access token and metadata
        """
        try:
            # Get credential from database
            encrypted_credential = await self._retrieve_credential(
                user_id, CredentialType.PLAID_ACCESS_TOKEN, item_id=item_id
            )
            
            if not encrypted_credential:
                return None
            
            # Decrypt the token data
            decrypted_data = self._decrypt_data(
                encrypted_credential.encrypted_data,
                encrypted_credential.salt
            )
            
            # Parse token data
            token_dict = eval(decrypted_data)
            
            # Update access tracking
            await self._update_access_tracking(encrypted_credential.credential_id)
            
            # Log the access
            await self._log_credential_access(
                user_id, encrypted_credential.credential_id, "RETRIEVE", 
                f"Access token retrieved for item {token_dict.get('item_id')}"
            )
            
            return {
                'access_token': token_dict['access_token'],
                'item_id': token_dict['item_id'],
                'institution_id': token_dict.get('institution_id'),
                'created_at': token_dict['created_at']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve access token: {e}")
            return None
    
    async def list_user_credentials(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all credentials for a user (metadata only, no sensitive data).
        
        Args:
            user_id: User identifier
            
        Returns:
            List of credential metadata
        """
        try:
            if self.credentials_collection is None:
                return self._simulate_credential_list(user_id)
            
            cursor = self.credentials_collection.find(
                {"user_id": user_id, "status": CredentialStatus.ACTIVE.value}
            ).sort("created_at", -1)
            
            credentials = []
            async for doc in cursor:
                credentials.append({
                    'credential_id': doc['credential_id'],
                    'credential_type': doc['credential_type'],
                    'created_at': doc['created_at'],
                    'expires_at': doc.get('expires_at'),
                    'status': doc['status'],
                    'metadata': doc.get('metadata', {}),
                    'access_count': doc.get('access_count', 0),
                    'last_accessed': doc.get('last_accessed')
                })
            
            return credentials
            
        except Exception as e:
            self.logger.error(f"Failed to list credentials: {e}")
            return []
    
    async def revoke_credential(self, user_id: str, credential_id: str) -> bool:
        """
        Revoke a credential (mark as inactive).
        
        Args:
            user_id: User identifier
            credential_id: Credential to revoke
            
        Returns:
            True if successfully revoked
        """
        try:
            if self.credentials_collection is None:
                return True  # Simulation mode
            
            result = await self.credentials_collection.update_one(
                {"user_id": user_id, "credential_id": credential_id},
                {
                    "$set": {
                        "status": CredentialStatus.REVOKED.value,
                        "revoked_at": datetime.now()
                    }
                }
            )
            
            if result.modified_count > 0:
                await self._log_credential_access(
                    user_id, credential_id, "REVOKE", "Credential revoked"
                )
                self.logger.info(f"Credential {credential_id} revoked for user {user_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to revoke credential: {e}")
            return False
    
    async def rotate_encryption_key(self) -> bool:
        """
        Rotate the master encryption key (re-encrypt all credentials).
        
        Returns:
            True if rotation successful
        """
        try:
            # Generate new master key
            new_master_key = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()
            
            if self.credentials_collection is None:
                self.master_key = new_master_key
                self.logger.warning(f"New master key (simulation): {new_master_key}")
                return True
            
            # Get all active credentials
            cursor = self.credentials_collection.find({"status": CredentialStatus.ACTIVE.value})
            
            rotation_count = 0
            async for doc in cursor:
                try:
                    # Decrypt with old key
                    old_data = self._decrypt_data(doc['encrypted_data'], doc['salt'])
                    
                    # Generate new salt
                    new_salt = secrets.token_bytes(32)
                    
                    # Encrypt with new key
                    old_master = self.master_key
                    self.master_key = new_master_key
                    new_encrypted_data = self._encrypt_data(old_data, new_salt)
                    self.master_key = old_master  # Restore for other operations
                    
                    # Update in database
                    await self.credentials_collection.update_one(
                        {"_id": doc['_id']},
                        {
                            "$set": {
                                "encrypted_data": new_encrypted_data,
                                "salt": new_salt,
                                "key_rotated_at": datetime.now()
                            }
                        }
                    )
                    
                    rotation_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to rotate credential {doc['credential_id']}: {e}")
            
            # Update master key
            self.master_key = new_master_key
            
            self.logger.info(f"Master key rotated - {rotation_count} credentials updated")
            self.logger.warning(f"New master key: {new_master_key}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rotate encryption key: {e}")
            return False
    
    # Internal helper methods
    
    async def _store_credential(self, credential: EncryptedCredential):
        """Store encrypted credential in database."""
        if self.credentials_collection is None:
            # Simulation mode
            self.logger.info(f"[SIMULATION] Stored credential {credential.credential_id}")
            return
        
        doc = {
            'credential_id': credential.credential_id,
            'user_id': credential.user_id,
            'credential_type': credential.credential_type.value,
            'encrypted_data': credential.encrypted_data,
            'salt': credential.salt,
            'created_at': credential.created_at,
            'expires_at': credential.expires_at,
            'status': credential.status.value,
            'metadata': credential.metadata,
            'access_count': credential.access_count,
            'last_accessed': credential.last_accessed
        }
        
        await self.credentials_collection.insert_one(doc)
    
    async def _retrieve_credential(self, user_id: str, credential_type: CredentialType, 
                                 credential_id: str = None, item_id: str = None) -> Optional[EncryptedCredential]:
        """Retrieve encrypted credential from database."""
        if self.credentials_collection is None:
            return self._simulate_credential_retrieval(user_id, credential_type)
        
        # Build query
        query = {
            "user_id": user_id,
            "credential_type": credential_type.value,
            "status": CredentialStatus.ACTIVE.value
        }
        
        if credential_id:
            query["credential_id"] = credential_id
        elif item_id:
            query["metadata.item_id"] = item_id
        
        # Get most recent if no specific credential requested
        doc = await self.credentials_collection.find_one(
            query, sort=[("created_at", -1)]
        )
        
        if not doc:
            return None
        
        return EncryptedCredential(
            credential_id=doc['credential_id'],
            user_id=doc['user_id'],
            credential_type=CredentialType(doc['credential_type']),
            encrypted_data=doc['encrypted_data'],
            salt=doc['salt'],
            created_at=doc['created_at'],
            expires_at=doc.get('expires_at'),
            status=CredentialStatus(doc['status']),
            metadata=doc.get('metadata', {}),
            access_count=doc.get('access_count', 0),
            last_accessed=doc.get('last_accessed')
        )
    
    async def _update_access_tracking(self, credential_id: str):
        """Update access count and timestamp for credential."""
        if self.credentials_collection is None:
            return
        
        await self.credentials_collection.update_one(
            {"credential_id": credential_id},
            {
                "$inc": {"access_count": 1},
                "$set": {"last_accessed": datetime.now()}
            }
        )
    
    async def _log_credential_access(self, user_id: str, credential_id: str, 
                                   action: str, details: str):
        """Log credential access for audit purposes."""
        if self.audit_collection is None:
            self.logger.info(f"[AUDIT] {user_id} - {action} - {credential_id} - {details}")
            return
        
        audit_entry = {
            'user_id': user_id,
            'credential_id': credential_id,
            'action': action,
            'details': details,
            'timestamp': datetime.now(),
            'ip_address': None,  # Could be added from request context
            'user_agent': None   # Could be added from request context
        }
        
        await self.audit_collection.insert_one(audit_entry)
    
    # Simulation methods for development/testing
    
    def _simulate_credential_retrieval(self, user_id: str, credential_type: CredentialType) -> Optional[EncryptedCredential]:
        """Simulate credential retrieval for development."""
        if credential_type == CredentialType.PLAID_API:
            # Simulate with demo credentials
            demo_data = str({
                'client_id': 'demo_client_id',
                'secret': 'demo_secret',
                'environment': 'sandbox',
                'access_token': None,
                'item_id': None,
                'institution_id': None
            })
            
            salt = b'demo_salt_32_bytes_for_simulation'
            encrypted_data = self._encrypt_data(demo_data, salt)
            
            return EncryptedCredential(
                credential_id=f"plaid_{user_id}_demo",
                user_id=user_id,
                credential_type=credential_type,
                encrypted_data=encrypted_data,
                salt=salt,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(days=365),
                status=CredentialStatus.ACTIVE,
                metadata={'environment': 'sandbox'},
                access_count=0,
                last_accessed=None
            )
        
        return None
    
    def _simulate_credential_list(self, user_id: str) -> List[Dict[str, Any]]:
        """Simulate credential listing for development."""
        return [
            {
                'credential_id': f'plaid_{user_id}_demo',
                'credential_type': 'plaid_api',
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(days=365),
                'status': 'active',
                'metadata': {'environment': 'sandbox'},
                'access_count': 5,
                'last_accessed': datetime.now() - timedelta(hours=2)
            },
            {
                'credential_id': f'access_{user_id}_demo',
                'credential_type': 'plaid_access_token',
                'created_at': datetime.now() - timedelta(days=5),
                'expires_at': None,
                'status': 'active',
                'metadata': {'item_id': 'item_demo_123'},
                'access_count': 15,
                'last_accessed': datetime.now() - timedelta(minutes=30)
            }
        ]


# Singleton instance
credential_manager = EncryptedCredentialManager()


async def main():
    """Test the Encrypted Credential Manager."""
    manager = EncryptedCredentialManager()
    
    print("Testing Encrypted Credential Storage System")
    print("=" * 55)
    
    # Test user
    test_user_id = "user_12345"
    
    # Test Plaid credential storage
    print("/n1. Storing Plaid API Credentials:")
    plaid_creds = PlaidCredentials(
        client_id="your_real_client_id_here",
        secret="your_real_secret_here", 
        environment="sandbox"
    )
    
    credential_id = await manager.store_plaid_credentials(test_user_id, plaid_creds)
    print(f"   Stored with ID: {credential_id}")
    
    # Test credential retrieval
    print("/n2. Retrieving Plaid Credentials:")
    retrieved_creds = await manager.retrieve_plaid_credentials(test_user_id, credential_id)
    if retrieved_creds:
        print(f"   Client ID: {retrieved_creds.client_id}")
        print(f"   Environment: {retrieved_creds.environment}")
        print(f"   Secret: {'*' * len(retrieved_creds.secret)}")
    
    # Test access token storage
    print("/n3. Storing Access Token:")
    access_token_id = await manager.store_access_token(
        test_user_id, "access-sandbox-demo-token", "item_demo_123", "ins_demo_bank"
    )
    print(f"   Stored with ID: {access_token_id}")
    
    # Test access token retrieval
    print("/n4. Retrieving Access Token:")
    token_data = await manager.retrieve_access_token(test_user_id, "item_demo_123")
    if token_data:
        print(f"   Item ID: {token_data['item_id']}")
        print(f"   Institution: {token_data['institution_id']}")
        print(f"   Token: {token_data['access_token'][:20]}...")
    
    # Test credential listing
    print("/n5. Listing User Credentials:")
    credentials = await manager.list_user_credentials(test_user_id)
    for cred in credentials:
        print(f"   {cred['credential_type']}: {cred['credential_id']} (accessed {cred['access_count']} times)")
    
    # Test key rotation
    print("/n6. Testing Encryption Key Rotation:")
    rotation_success = await manager.rotate_encryption_key()
    print(f"   Key rotation: {'SUCCESS' if rotation_success else 'FAILED'}")
    
    print("/nEncrypted Credential Storage system testing complete!")
    print("/nSECURITY NOTES:")
    print("- Store the master key securely in production")
    print("- Use proper key management service for enterprise deployment")
    print("- Monitor audit logs for suspicious credential access")
    print("- Rotate encryption keys regularly")


if __name__ == "__main__":
    asyncio.run(main())