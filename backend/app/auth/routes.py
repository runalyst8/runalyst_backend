from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.deps.auth import get_current_user
from app.deps.db import get_db
from app.models.user import User
from app.models.profile_info import ProfileInfo
from app.auth.schemas import SignUpIn, TokenOut, UserOut, ProfileUpdateIn, ProfileOut
from app.auth.schemas import PasswordResetRequestIn, PasswordResetIn, SendVerificationIn, VerifyEmailIn
from app.core.security import hash_password, verify_password, create_access_token, decode_access_token
from app.core.security import create_password_reset_token, decode_password_reset_token
from app.services.storage import supabase_client
from app.services import otp_store, email as email_service

router = APIRouter(prefix="/auth", tags=["auth"])
bearer = HTTPBearer(auto_error=False)


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(payload: SignUpIn, db: Session = Depends(get_db)):
    # Validate password length (bcrypt limit)
    if len(payload.password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot exceed 72 bytes"
        )
    
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    try:
        user = User(email=payload.email, hashed_password=hash_password(payload.password))
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # FIX: Return the ORM object directly. Pydantic's 'from_attributes = True' 
        # (formerly from_orm) will automatically map all fields (id, email, is_active, created_at)
        return user
        
    except Exception as e:
        db.rollback()
        # Log the exception for debugging on your side
        print(f"Error during user creation: {e}") 
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.post("/login", response_model=TokenOut)
def login(payload: SignUpIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please verify your email before signing in."
        )
    token = create_access_token(sub=str(user.id))
    return TokenOut(access_token=token)


@router.post("/send-verification-email", status_code=status.HTTP_200_OK)
def send_verification_email(payload: SendVerificationIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        # Don't reveal whether the email exists
        return {"detail": "If that email is registered, a verification code has been sent."}
    if user.is_verified:
        return {"detail": "Email is already verified."}
    code = otp_store.save_otp(payload.email)
    try:
        email_service.send_verification_email(payload.email, code)
    except Exception as e:
        print(f"Email send error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email. Please try again."
        )
    return {"detail": "Verification code sent."}


@router.post("/verify-email", status_code=status.HTTP_200_OK)
def verify_email(payload: VerifyEmailIn, db: Session = Depends(get_db)):
    if not otp_store.verify_otp(payload.email, payload.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code."
        )
    user = db.query(User).filter(User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found.")
    user.is_verified = True
    db.commit()
    return {"detail": "Email verified successfully."}

@router.get("/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user

@router.post("/test")
def test_endpoint():
    return {"msg": "Test endpoint is working!"}

@router.post("/request-password-reset")
def request_password_reset(
        payload: PasswordResetRequestIn,
        db: Session = Depends(get_db)
):
    try:
        user = db.query(User).filter(User.email == payload.email).first()
    except Exception as e:
        print(f"Database error during password reset request: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User email cannot found"
        )
    if not user:
        # Note: We don't want to reveal if an email exists or not
        # for security reasons. So we return a generic success message.
        return {"msg": "If a user with that email exists, a password reset link has been sent."}

    try:
        password_reset_token = create_password_reset_token(email=user.email)
    except Exception as e:
        print(f"Error creating password reset token: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create password reset token"
        )

    # In a real application, you would send the token via email here.
    # For this example, we'll just print it.
    print(f"Password reset token for {user.email}: {password_reset_token}")

    return {"msg": "If a user with that email exists, a password reset link has been sent."}


@router.post("/reset-password")
def reset_password(
        payload: PasswordResetIn,
        db: Session = Depends(get_db)
):
    email = decode_password_reset_token(payload.token)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired token"
        )

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Validate new password length
    if len(payload.new_password.encode('utf-8')) > 72:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password cannot exceed 72 bytes"
        )

    user.hashed_password = hash_password(payload.new_password)
    db.add(user)
    db.commit()

    return {"msg": "Password updated successfully"}


#placeholder account deletion, no email confirmation needed like signup
#will be updated in future together along with signup to include email confirmation flows
@router.delete("/me", status_code=status.HTTP_200_OK)
def delete_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        db.delete(current_user)
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Error during user deletion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )
    return {"detail": "Account deleted successfully"}


@router.post("/generate-upload-url", status_code=status.HTTP_200_OK)
def create_upload_url(current_user: User = Depends(get_current_user)):
    bucket_name = "user_videos_test"
    unique_filename = f"{current_user.id}/{uuid.uuid4()}.mp4"

    try:
        signed_url_response = supabase_client.storage.from_(bucket_name).create_signed_upload_url(
            path=unique_filename
        )
        #print("DEBUG: Supabase response:", signed_url_response)
        return {
            "upload_url": signed_url_response['signed_url'],
            "path": signed_url_response['path']
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create upload URL: {str(e)}"
        )


@router.patch("/update-profile", response_model=ProfileOut)
def update_profile(
        payload: ProfileUpdateIn,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_db)
):
    # 1. Try to find existing profile for this user
    profile = db.query(ProfileInfo).filter(ProfileInfo.user_id == current_user.id).first()

    # 2. Extract the data sent in the request
    # We want Enum objects, not strings, for SQLAlchemy Enum columns
    update_data = payload.model_dump(exclude_unset=True)

    try:
        if not profile:
            # Create new profile if it doesn't exist
            profile = ProfileInfo(user_id=current_user.id, **update_data)
            db.add(profile)
        else:
            # Update only the fields provided in the payload
            for key, value in update_data.items():
                setattr(profile, key, value)

        db.commit()
        db.refresh(profile)
        return profile

    except Exception as e:
        db.rollback()
        print(f"Error updating profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile information"
        )