"""
Database Table Cleaner Script
Removes all content from existing tables in the Enhanced EHR database.
Handles foreign key constraints by deleting in proper order.
"""

import os
import sys
import logging
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text, inspect

# Add the app directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.enhanced_ehr_schema import (
        SessionLocal, engine, Base,
        PatientSummary, ClinicalNote, CarePlan, Procedure, Observation,
        AllergyIntolerance, MedicationStatement, EncounterDiagnosis, 
        Condition, Encounter, PatientIdentifier, Patient, Provider
    )
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the correct directory and enhanced_ehr_schema.py exists")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseCleaner:
    """Clean all data from Enhanced EHR database tables."""
    
    def __init__(self):
        self.db: Session = None
        
        # Define tables in deletion order (reverse dependency order)
        # Child tables first, then parent tables
        self.table_deletion_order = [
            # Level 4: Tables that reference other tables heavily
            PatientSummary,
            ClinicalNote,
            
            # Level 3: Tables that reference encounters, patients, providers
            CarePlan,
            Procedure,
            Observation,
            AllergyIntolerance,
            MedicationStatement,
            EncounterDiagnosis,
            
            # Level 2: Tables that reference patients and providers
            Condition,
            Encounter,
            PatientIdentifier,
            
            # Level 1: Base tables with minimal dependencies
            Patient,
            Provider
        ]
    
    def connect_to_database(self):
        """Establish database connection."""
        try:
            self.db = SessionLocal()
            logger.info("✅ Connected to database successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def get_table_stats(self) -> dict:
        """Get current record counts for all tables."""
        stats = {}
        
        if not self.db:
            return stats
        
        try:
            for table_class in self.table_deletion_order:
                table_name = table_class.__tablename__
                count = self.db.query(table_class).count()
                stats[table_name] = count
            
            return stats
        except Exception as e:
            logger.error(f"❌ Error getting table stats: {e}")
            return {}
    
    def print_table_stats(self, stats: dict, title: str):
        """Print formatted table statistics."""
        logger.info("=" * 60)
        logger.info(f"📊 {title}")
        logger.info("=" * 60)
        
        total_records = 0
        for table_name, count in stats.items():
            logger.info(f"{table_name:25} | {count:8,} records")
            total_records += count
        
        logger.info("-" * 60)
        logger.info(f"{'TOTAL':25} | {total_records:8,} records")
        logger.info("=" * 60)
    
    def disable_foreign_key_checks(self):
        """Disable foreign key constraints for faster deletion."""
        try:
            # Check database type
            inspector = inspect(engine)
            dialect_name = engine.dialect.name
            
            if dialect_name == 'sqlite':
                self.db.execute(text("PRAGMA foreign_keys = OFF"))
                logger.info("🔧 Disabled foreign key constraints (SQLite)")
            elif dialect_name == 'postgresql':
                # For PostgreSQL, we'll handle this differently by deleting in order
                logger.info("🔧 PostgreSQL detected - will delete in dependency order")
            elif dialect_name == 'mysql':
                self.db.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
                logger.info("🔧 Disabled foreign key constraints (MySQL)")
            
            self.db.commit()
        except Exception as e:
            logger.warning(f"⚠️ Could not disable foreign key constraints: {e}")
    
    def enable_foreign_key_checks(self):
        """Re-enable foreign key constraints."""
        try:
            dialect_name = engine.dialect.name
            
            if dialect_name == 'sqlite':
                self.db.execute(text("PRAGMA foreign_keys = ON"))
                logger.info("🔧 Re-enabled foreign key constraints (SQLite)")
            elif dialect_name == 'mysql':
                self.db.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
                logger.info("🔧 Re-enabled foreign key constraints (MySQL)")
            
            self.db.commit()
        except Exception as e:
            logger.warning(f"⚠️ Could not re-enable foreign key constraints: {e}")
    
    def clear_table(self, table_class) -> int:
        """Clear all records from a specific table."""
        table_name = table_class.__tablename__
        
        try:
            # Get count before deletion
            initial_count = self.db.query(table_class).count()
            
            if initial_count == 0:
                logger.info(f"⏭️ Table '{table_name}' is already empty")
                return 0
            
            # Delete all records
            deleted_count = self.db.query(table_class).delete()
            
            logger.info(f"🗑️ Cleared {deleted_count:,} records from '{table_name}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error clearing table '{table_name}': {e}")
            return 0
    
    def clear_all_tables(self) -> dict:
        """Clear all tables in the correct order."""
        deletion_stats = {}
        total_deleted = 0
        
        logger.info("🧹 Starting database cleanup...")
        
        # Disable foreign key constraints for faster deletion
        self.disable_foreign_key_checks()
        
        try:
            # Delete tables in dependency order
            for table_class in self.table_deletion_order:
                table_name = table_class.__tablename__
                deleted_count = self.clear_table(table_class)
                deletion_stats[table_name] = deleted_count
                total_deleted += deleted_count
            
            # Commit all deletions
            self.db.commit()
            logger.info("✅ All deletions committed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during table clearing: {e}")
            self.db.rollback()
            logger.info("🔄 Database rollback completed")
            raise
        
        finally:
            # Re-enable foreign key constraints
            self.enable_foreign_key_checks()
        
        deletion_stats['_total_deleted'] = total_deleted
        return deletion_stats
    
    def reset_auto_increment(self):
        """Reset auto-increment counters for tables that use them."""
        try:
            dialect_name = engine.dialect.name
            
            if dialect_name == 'sqlite':
                # SQLite doesn't need explicit reset for autoincrement
                logger.info("🔄 SQLite auto-increment will reset automatically")
                
            elif dialect_name == 'postgresql':
                # Reset sequences for tables with SERIAL columns
                tables_with_sequences = [
                    ('patient_identifiers', 'patient_identifiers_id_seq'),
                    ('encounter_diagnoses', 'encounter_diagnoses_id_seq')
                ]
                
                for table_name, sequence_name in tables_with_sequences:
                    try:
                        self.db.execute(text(f"ALTER SEQUENCE {sequence_name} RESTART WITH 1"))
                        logger.info(f"🔄 Reset sequence for {table_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not reset sequence for {table_name}: {e}")
                
            elif dialect_name == 'mysql':
                # Reset auto-increment for MySQL tables
                tables_with_auto_increment = [
                    'patient_identifiers',
                    'encounter_diagnoses'
                ]
                
                for table_name in tables_with_auto_increment:
                    try:
                        self.db.execute(text(f"ALTER TABLE {table_name} AUTO_INCREMENT = 1"))
                        logger.info(f"🔄 Reset auto-increment for {table_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not reset auto-increment for {table_name}: {e}")
            
            self.db.commit()
            
        except Exception as e:
            logger.warning(f"⚠️ Could not reset auto-increment counters: {e}")
    
    def verify_cleanup(self) -> bool:
        """Verify that all tables are empty after cleanup."""
        try:
            stats = self.get_table_stats()
            total_remaining = sum(stats.values())
            
            if total_remaining == 0:
                logger.info("✅ Database cleanup verification successful - all tables are empty")
                return True
            else:
                logger.error(f"❌ Database cleanup verification failed - {total_remaining} records remain")
                self.print_table_stats(stats, "REMAINING RECORDS AFTER CLEANUP")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during cleanup verification: {e}")
            return False
    
    def close_connection(self):
        """Close database connection."""
        if self.db:
            self.db.close()
            logger.info("🔒 Database connection closed")

def confirm_cleanup() -> bool:
    """Ask user for confirmation before cleaning database."""
    print("\n" + "="*60)
    print("⚠️  DATABASE CLEANUP CONFIRMATION")
    print("="*60)
    print("🚨 WARNING: This will permanently delete ALL data from the database!")
    print("📊 This includes all patients, encounters, medications, notes, etc.")
    print("💾 Make sure you have backups if you need to preserve any data.")
    print("="*60)
    
    response = input("Are you sure you want to proceed? Type 'YES' to continue: ").strip()
    
    if response.upper() == 'YES':
        print("✅ Cleanup confirmed. Proceeding...")
        return True
    else:
        print("❌ Cleanup cancelled.")
        return False

def main():
    """Main function to run the database cleanup."""
    print("🧹 Enhanced EHR Database Cleanup Utility")
    print("="*60)
    
    # Initialize cleaner
    cleaner = DatabaseCleaner()
    
    try:
        # Connect to database
        if not cleaner.connect_to_database():
            sys.exit(1)
        
        # Get initial statistics
        initial_stats = cleaner.get_table_stats()
        total_initial = sum(initial_stats.values())
        
        if total_initial == 0:
            logger.info("ℹ️ Database is already empty. No cleanup needed.")
            return
        
        cleaner.print_table_stats(initial_stats, "CURRENT DATABASE CONTENTS")
        
        # Ask for confirmation
        if not confirm_cleanup():
            return
        
        # Perform cleanup
        deletion_stats = cleaner.clear_all_tables()
        
        # Reset auto-increment counters
        cleaner.reset_auto_increment()
        
        # Verify cleanup
        verification_success = cleaner.verify_cleanup()
        
        # Final statistics
        final_stats = cleaner.get_table_stats()
        total_deleted = deletion_stats.get('_total_deleted', 0)
        
        print("\n" + "="*60)
        print("📋 DATABASE CLEANUP SUMMARY")
        print("="*60)
        print(f"📊 Initial records: {total_initial:,}")
        print(f"🗑️ Records deleted: {total_deleted:,}")
        print(f"📊 Final records: {sum(final_stats.values()):,}")
        print(f"✅ Verification: {'PASSED' if verification_success else 'FAILED'}")
        print("="*60)
        
        if verification_success:
            print("🎉 Database cleanup completed successfully!")
            print("🚀 Ready for fresh data insertion.")
        else:
            print("⚠️ Database cleanup completed with warnings.")
            print("🔍 Check the logs above for details.")
        
    except KeyboardInterrupt:
        print("\n❌ Cleanup cancelled by user.")
    except Exception as e:
        logger.error(f"❌ Unexpected error during cleanup: {e}")
        sys.exit(1)
    finally:
        cleaner.close_connection()

if __name__ == "__main__":
    main()