from hypatia.backend.constraints.Balance import Balance
from hypatia.backend.constraints.BalanceUnMetDemand import BalanceUnMetDemand
from hypatia.backend.constraints.TradeBalance import TradeBalance
from hypatia.backend.constraints.ResourceTechAvailability import ResourceTechAvailability
from hypatia.backend.constraints.TotalCapacityRegional import TotalCapacityRegional
from hypatia.backend.constraints.NewCapacityRegional import NewCapacityRegional
from hypatia.backend.constraints.TechEfficency import TechEfficency
from hypatia.backend.constraints.AnnualProductionRegional import AnnualProductionRegional
from hypatia.backend.constraints.EmissionCapRegional import EmissionCapRegional
from hypatia.backend.constraints.EmissionCapGlobal import EmissionCapGlobal
from hypatia.backend.constraints.EmissionConsumedRegional import EmissionConsumedRegional
from hypatia.backend.constraints.StorageMaxMinChange import StorageMaxMinChange
from hypatia.backend.constraints.StorageMaxFlowInOut import StorageMaxFlowInOut
from hypatia.backend.constraints.StorageCyclicBoundary import StorageCyclicBoundary
from hypatia.backend.constraints.LineTotalCapacity import LineTotalCapacity
from hypatia.backend.constraints.TotalCapacityGlobal import TotalCapacityGlobal
from hypatia.backend.constraints.NewCapacityGlobal import NewCapacityGlobal
from hypatia.backend.constraints.AnnualProductionGlobal import AnnualProductionGlobal
from hypatia.backend.constraints.LineAvailability import LineAvailability
from hypatia.backend.constraints.LineNewCapacity import LineNewCapacity
from hypatia.backend.constraints.LandUsageRegional import LandUsageRegional
from hypatia.backend.constraints.LandUsageGlobal import LandUsageGlobal
from hypatia.backend.constraints.RenewableProductionRegional import RenewableProductionRegional
from hypatia.backend.constraints.RenewableProductionGlobal import RenewableProductionGlobal  
# from hypatia.backend.constraints.ElectrolysisConsumption import ElectrolysisConsumption # To make it work, "Electricity" carrier must be explicitly specified

#from hypatia.backend.constraints.NewCapacityNonneg import NewCapacityNonneg # This constraint is functional to MILP version, but it is not implemented in the current model, so it is not included in the list of constraints.
#from hypatia.backend.constraints.ProductionRamp import ProductionRamp # The current definition of ProductionRamp is not compatible with the effects brought by the StorageCyclciBoundary constraint, so it is not included in the list of constraints.

CONSTRAINTS = [
    Balance,
    BalanceUnMetDemand,
    TradeBalance,
    ResourceTechAvailability,
    TotalCapacityRegional,
    NewCapacityRegional,
    TechEfficency,
    AnnualProductionRegional,
    EmissionCapRegional,
    EmissionCapGlobal,
    EmissionConsumedRegional,
    StorageMaxMinChange,
    StorageMaxFlowInOut,
    StorageCyclicBoundary,
    LineTotalCapacity,
    TotalCapacityGlobal,
    NewCapacityGlobal,
    AnnualProductionGlobal,
    RenewableProductionGlobal,
    LineAvailability,
    LineNewCapacity,
    RenewableProductionRegional,
    LandUsageRegional,
    LandUsageGlobal,
    # ElectrolysisConsumption
]


