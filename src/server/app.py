"""Federated Learning Server — FastAPI skeleton.

This module provides:
- FLServer: core server state and orchestration logic
- FastAPI endpoints for client registration, round management, health checks
- In-memory client registry and round state

Run with: uvicorn src.server.app:app --reload
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel


# ---------------------- Data Models ----------------------

class ClientInfo(BaseModel):
    """Client registration info."""
    client_id: str
    num_samples: int
    metadata: Optional[Dict] = None


class ClientRegistration(BaseModel):
    """Response after successful registration."""
    client_id: str
    registered_at: str
    message: str


class RoundStatus(BaseModel):
    """Current round status."""
    round_id: int
    state: str  # "idle", "in_progress", "completed"
    participants: List[str]
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ServerStatus(BaseModel):
    """Server health and status."""
    status: str
    registered_clients: int
    current_round: int
    round_state: str
    uptime_seconds: float


class RoundConfig(BaseModel):
    """Configuration for starting a new round."""
    min_clients: int = 2
    max_clients: Optional[int] = None
    timeout_seconds: int = 300


# ---------------------- Server State ----------------------

@dataclass
class FLServer:
    """Core federated learning server state and logic.

    Thread-safe via a simple lock. For production, use proper async patterns.
    """

    clients: Dict[str, Dict] = field(default_factory=dict)
    current_round: int = 0
    round_state: str = "idle"  # "idle", "in_progress", "completed"
    round_participants: List[str] = field(default_factory=list)
    round_started_at: Optional[datetime] = None
    round_completed_at: Optional[datetime] = None
    started_at: datetime = field(default_factory=datetime.utcnow)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def register_client(self, client_id: str, num_samples: int, metadata: Optional[Dict] = None) -> Dict:
        """Register a new client or update existing."""
        with self._lock:
            now = datetime.utcnow()
            self.clients[client_id] = {
                "client_id": client_id,
                "num_samples": num_samples,
                "metadata": metadata or {},
                "registered_at": now.isoformat(),
                "last_seen": now.isoformat(),
            }
            return self.clients[client_id]

    def unregister_client(self, client_id: str) -> bool:
        """Remove a client from registry."""
        with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
                return True
            return False

    def get_client(self, client_id: str) -> Optional[Dict]:
        """Get client info by ID."""
        with self._lock:
            return self.clients.get(client_id)

    def list_clients(self) -> List[Dict]:
        """List all registered clients."""
        with self._lock:
            return list(self.clients.values())

    def start_round(self, participant_ids: Optional[List[str]] = None) -> RoundStatus:
        """Start a new training round."""
        with self._lock:
            if self.round_state == "in_progress":
                raise ValueError("A round is already in progress")

            self.current_round += 1
            self.round_state = "in_progress"
            self.round_started_at = datetime.utcnow()
            self.round_completed_at = None

            # select participants (all clients if none specified)
            if participant_ids is None:
                self.round_participants = list(self.clients.keys())
            else:
                self.round_participants = [p for p in participant_ids if p in self.clients]

            return RoundStatus(
                round_id=self.current_round,
                state=self.round_state,
                participants=self.round_participants,
                started_at=self.round_started_at.isoformat() if self.round_started_at else None,
            )

    def complete_round(self) -> RoundStatus:
        """Mark current round as completed."""
        with self._lock:
            if self.round_state != "in_progress":
                raise ValueError("No round in progress")

            self.round_state = "completed"
            self.round_completed_at = datetime.utcnow()

            return RoundStatus(
                round_id=self.current_round,
                state=self.round_state,
                participants=self.round_participants,
                started_at=self.round_started_at.isoformat() if self.round_started_at else None,
                completed_at=self.round_completed_at.isoformat() if self.round_completed_at else None,
            )

    def reset_round(self) -> None:
        """Reset to idle state (e.g., after timeout or error)."""
        with self._lock:
            self.round_state = "idle"
            self.round_participants = []

    def get_status(self) -> ServerStatus:
        """Get server status."""
        with self._lock:
            uptime = (datetime.utcnow() - self.started_at).total_seconds()
            return ServerStatus(
                status="healthy",
                registered_clients=len(self.clients),
                current_round=self.current_round,
                round_state=self.round_state,
                uptime_seconds=uptime,
            )


# ---------------------- FastAPI App ----------------------

app = FastAPI(
    title="Federated Learning Server",
    description="Server for orchestrating federated predictive maintenance training",
    version="0.1.0",
)

# Global server instance
server = FLServer()


# ---------------------- Health & Status Endpoints ----------------------

@app.get("/health", response_model=ServerStatus, tags=["Health"])
def health_check():
    """Health check endpoint."""
    return server.get_status()


@app.get("/status", response_model=ServerStatus, tags=["Health"])
def get_status():
    """Get detailed server status."""
    return server.get_status()


# ---------------------- Client Registration Endpoints ----------------------

@app.post("/clients/register", response_model=ClientRegistration, tags=["Clients"])
def register_client(info: ClientInfo):
    """Register a new client or update existing registration."""
    result = server.register_client(info.client_id, info.num_samples, info.metadata)
    return ClientRegistration(
        client_id=result["client_id"],
        registered_at=result["registered_at"],
        message="Client registered successfully",
    )


@app.delete("/clients/{client_id}", tags=["Clients"])
def unregister_client(client_id: str):
    """Unregister a client."""
    if server.unregister_client(client_id):
        return {"message": f"Client {client_id} unregistered"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")


@app.get("/clients", response_model=List[Dict], tags=["Clients"])
def list_clients():
    """List all registered clients."""
    return server.list_clients()


@app.get("/clients/{client_id}", tags=["Clients"])
def get_client(client_id: str):
    """Get info for a specific client."""
    client = server.get_client(client_id)
    if client is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Client not found")
    return client


# ---------------------- Round Orchestration Endpoints ----------------------

@app.get("/rounds/current", response_model=RoundStatus, tags=["Rounds"])
def get_current_round():
    """Get current round status."""
    return RoundStatus(
        round_id=server.current_round,
        state=server.round_state,
        participants=server.round_participants,
        started_at=server.round_started_at.isoformat() if server.round_started_at else None,
        completed_at=server.round_completed_at.isoformat() if server.round_completed_at else None,
    )


@app.post("/rounds/start", response_model=RoundStatus, tags=["Rounds"])
def start_round(config: Optional[RoundConfig] = None):
    """Start a new training round."""
    try:
        # check minimum clients
        if config and len(server.clients) < config.min_clients:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not enough clients. Need {config.min_clients}, have {len(server.clients)}",
            )
        return server.start_round()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@app.post("/rounds/complete", response_model=RoundStatus, tags=["Rounds"])
def complete_round():
    """Mark the current round as completed."""
    try:
        return server.complete_round()
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@app.post("/rounds/reset", tags=["Rounds"])
def reset_round():
    """Reset round state to idle."""
    server.reset_round()
    return {"message": "Round state reset to idle"}
