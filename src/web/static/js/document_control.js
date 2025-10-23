/**
 * Global Document Control System
 * Provides unified document ID generation, tracking, and management
 */

class DocumentControlSystem {
    constructor() {
        this.documents = new Map();
        this.initializeSystem();
    }

    initializeSystem() {
        // Load existing documents from localStorage
        const stored = localStorage.getItem('document_registry');
        if (stored) {
            try {
                const data = JSON.parse(stored);
                this.documents = new Map(data);
            } catch (e) {
                console.warn('Failed to load document registry:', e);
            }
        }
    }

    /**
     * Generate a standardized document ID
     * @param {string} type - Document type (RPT, AUT, SCR, KNW, etc.)
     * @param {string} subtype - Document subtype/category
     * @param {string|null} sourceId - Original source ID if available
     * @param {Date|null} timestamp - Creation timestamp
     * @returns {string} Formatted document ID
     */
    generateDocumentId(type, subtype = 'GEN', sourceId = null, timestamp = null) {
        const date = timestamp ? new Date(timestamp) : new Date();
        const dateStr = date.toISOString().slice(0, 10).replace(/-/g, '');
        const timeStr = date.toTimeString().slice(0, 8).replace(/:/g, '');
        
        let suffix;
        if (sourceId) {
            suffix = sourceId.slice(-4).toUpperCase().padStart(4, '0');
        } else {
            suffix = Math.random().toString(36).substr(2, 4).toUpperCase();
        }
        
        return `${type}-${subtype}-${dateStr}-${timeStr}-${suffix}`;
    }

    /**
     * Register a document in the system
     * @param {object} document - Document metadata
     * @returns {string} Document ID
     */
    registerDocument(document) {
        const docId = this.generateDocumentId(
            document.type,
            document.subtype,
            document.sourceId,
            document.timestamp
        );

        const docRecord = {
            id: docId,
            type: document.type,
            subtype: document.subtype,
            title: document.title || 'Untitled',
            description: document.description || '',
            sourceId: document.sourceId,
            created: document.timestamp || new Date().toISOString(),
            page: document.page || window.location.pathname,
            status: document.status || 'active',
            metadata: document.metadata || {},
            tags: document.tags || []
        };

        this.documents.set(docId, docRecord);
        this.saveRegistry();
        
        console.log('Document registered:', docId, docRecord);
        return docId;
    }

    /**
     * Update document metadata
     */
    updateDocument(docId, updates) {
        if (this.documents.has(docId)) {
            const doc = this.documents.get(docId);
            Object.assign(doc, updates);
            doc.modified = new Date().toISOString();
            this.documents.set(docId, doc);
            this.saveRegistry();
            return true;
        }
        return false;
    }

    /**
     * Get document by ID
     */
    getDocument(docId) {
        return this.documents.get(docId);
    }

    /**
     * Search documents
     */
    searchDocuments(query) {
        const results = [];
        const normalizedQuery = query.toLowerCase();
        
        for (const [id, doc] of this.documents) {
            if (
                doc.title.toLowerCase().includes(normalizedQuery) ||
                doc.description.toLowerCase().includes(normalizedQuery) ||
                doc.id.toLowerCase().includes(normalizedQuery) ||
                doc.tags.some(tag => tag.toLowerCase().includes(normalizedQuery))
            ) {
                results.push(doc);
            }
        }
        
        return results.sort((a, b) => new Date(b.created) - new Date(a.created));
    }

    /**
     * Get documents by type
     */
    getDocumentsByType(type, subtype = null) {
        const results = [];
        for (const [id, doc] of this.documents) {
            if (doc.type === type && (!subtype || doc.subtype === subtype)) {
                results.push(doc);
            }
        }
        return results.sort((a, b) => new Date(b.created) - new Date(a.created));
    }

    /**
     * Export documents as CSV
     */
    exportDocuments(type = null) {
        let docs = Array.from(this.documents.values());
        if (type) {
            docs = docs.filter(doc => doc.type === type);
        }

        const headers = ['Document ID', 'Type', 'Subtype', 'Title', 'Created', 'Page', 'Status'];
        const csvContent = [
            headers.join(','),
            ...docs.map(doc => [
                `"${doc.id}"`,
                `"${doc.type}"`,
                `"${doc.subtype}"`,
                `"${doc.title}"`,
                `"${doc.created}"`,
                `"${doc.page}"`,
                `"${doc.status}"`
            ].join(','))
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `document_registry_${type || 'all'}_${new Date().toISOString().slice(0, 10)}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }

    /**
     * Create a document badge element
     */
    createDocumentBadge(docId, options = {}) {
        const doc = this.getDocument(docId);
        if (!doc) return null;

        const badge = document.createElement('code');
        badge.className = 'document-id-badge';
        badge.textContent = docId;
        badge.style.cssText = `
            background: linear-gradient(45deg, #007bff, #0056b3);
            color: white;
            padding: 0.25rem 0.5rem;
            border-radius: 0.375rem;
            font-size: 0.75rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-block;
            font-family: 'Courier New', monospace;
            border: none;
            margin-right: 0.5rem;
        `;

        badge.title = `${doc.title} - Click to copy`;
        badge.onclick = () => this.copyDocumentId(docId);
        
        badge.onmouseover = function() {
            this.style.background = 'linear-gradient(45deg, #0056b3, #004085)';
            this.style.transform = 'scale(1.05)';
            this.style.boxShadow = '0 2px 8px rgba(0, 123, 255, 0.3)';
        };
        
        badge.onmouseout = function() {
            this.style.background = 'linear-gradient(45deg, #007bff, #0056b3)';
            this.style.transform = 'scale(1)';
            this.style.boxShadow = 'none';
        };

        return badge;
    }

    /**
     * Copy document ID to clipboard
     */
    copyDocumentId(docId) {
        navigator.clipboard.writeText(docId).then(() => {
            if (window.RSApp) {
                RSApp.showMessage(`Document ID copied: ${docId}`, 'success');
            }
            console.log('Document ID copied:', docId);
        }).catch(() => {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = docId;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            if (window.RSApp) {
                RSApp.showMessage(`Document ID copied: ${docId}`, 'success');
            }
        });
    }

    /**
     * Save registry to localStorage
     */
    saveRegistry() {
        try {
            localStorage.setItem('document_registry', JSON.stringify([...this.documents]));
        } catch (e) {
            console.warn('Failed to save document registry:', e);
        }
    }

    /**
     * Get document statistics
     */
    getStatistics() {
        const docs = Array.from(this.documents.values());
        const stats = {
            total: docs.length,
            byType: {},
            recentCount: 0,
            oldestDate: null,
            newestDate: null
        };

        const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);

        docs.forEach(doc => {
            // Count by type
            const typeKey = `${doc.type}-${doc.subtype}`;
            stats.byType[typeKey] = (stats.byType[typeKey] || 0) + 1;

            // Count recent documents
            if (new Date(doc.created) > weekAgo) {
                stats.recentCount++;
            }

            // Track date range
            const docDate = new Date(doc.created);
            if (!stats.oldestDate || docDate < stats.oldestDate) {
                stats.oldestDate = docDate;
            }
            if (!stats.newestDate || docDate > stats.newestDate) {
                stats.newestDate = docDate;
            }
        });

        return stats;
    }
}

// Global document type definitions
const DOCUMENT_TYPES = {
    RPT: 'Reports',
    AUT: 'Automation Ideas', 
    SCR: 'Web Scrapes',
    KNW: 'Knowledge Entities',
    SYS: 'System Generated',
    USR: 'User Content'
};

const DOCUMENT_SUBTYPES = {
    // Report subtypes
    COMP: 'Company Analysis',
    TECH: 'Technology Report',
    MRKT: 'Market Analysis',
    COMP_INTEL: 'Competitive Intelligence',
    
    // Automation subtypes
    PROD: 'Productivity',
    WKFL: 'Workflow',
    INTG: 'Integration',
    
    // Scrape subtypes
    NEWS: 'News Articles',
    JOBS: 'Job Listings',
    PROD_DATA: 'Product Data',
    
    // Knowledge subtypes
    ENTS: 'Entities',
    RELS: 'Relationships',
    
    // General
    GEN: 'General'
};

// Initialize global document control system
window.DocumentControl = new DocumentControlSystem();

// Helper functions for easy access
window.registerDocument = (doc) => window.DocumentControl.registerDocument(doc);
window.createDocumentBadge = (docId) => window.DocumentControl.createDocumentBadge(docId);
window.searchDocuments = (query) => window.DocumentControl.searchDocuments(query);