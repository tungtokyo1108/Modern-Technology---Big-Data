/**
 * Google - Big Data Technology
 * https://github.com/googleapis/google-cloud-java/blob/master/google-cloud-clients/google-cloud-bigquery/src/main/java/com/google/cloud/bigquery/InsertAllRequest.java
 *
 *  Created on: Feb 22, 2019
 *  Student (MIG Virtual Developer): Tung Dang
 */

package com.google.cloud.bigquery.mirror;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.api.services.bigquery.Bigquery.Tabledata.InsertAll;
import com.google.api.services.bigquery.model.Table;
import com.google.cloud.bigquery.TableInfo;
import com.google.cloud.bigquery.InsertAllRequest.RowToInsert;
import com.google.cloud.bigquery.mirror.DatasetInfo.BuilderImpl;
import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.Lists;
import com.google.rpc.RetryInfo;

import java.io.Serializable;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class InsertAllRequest implements Serializable 
{
    private static final long serialVersionUID = 211200307773853078L;

    private final TableId table;
    private final List<RowToInsert> rows;
    private final Boolean skipInvalidRows;
    private final Boolean ignoreUnknownValues;
    private final String templateSuffix;

    public static class RowToInsert implements Serializable
    {
        private static final long serialVersionUID = 8563060538219179157L;

        private final String id;
        private final Map<String, Object> content;

        RowToInsert(String id, Map<String, ?> content)
        {
            this.id = id;
            if (content instanceof ImmutableBiMap)
            {
                this.content = ImmutableBiMap.copyOf(content);
            }
            else 
            {
                this.content = Collections.unmodifiableMap(new HashMap<>(content));
            }
        }

        public String getId()
        {
            return id;
        }

        public Map<String, Object> getContent()
        {
            return content;
        }

        @Override
        public String toString()
        {
            return MoreObjects.toStringHelper(this).add("id", id).add("content", content).toString();
        }

        @Override
        public int hashCode()
        {
            return Objects.hash(id, content);
        }

        @Override
        public boolean equals(Object obj)
        {
            if (!(obj instanceof RowToInsert)) {
                return false;
            }
            RowToInsert other = (RowToInsert) obj;
            return Objects.equals(id, other.id) && Objects.equals(content, other.content);
        }

        public static RowToInsert of(String id, Map<String, ?> content)
        {
            return new RowToInsert(checkNotNull(id), checkNotNull(content));
        }

        public static RowToInsert of(Map<String, ?> content)
        {
            return new RowToInsert(null, checkNotNull(content));
        }
    }

    public static final class Builder 
    {
        private TableId table;
        private List<RowToInsert> rows;
        private Boolean skipInvalidRows;
        private Boolean ignoreUnknownValues;
        private String templateSuffix;

        private Builder() {}

        public Builder setTable(TableId table)
        {
            this.table = checkNotNull(table);
            return this;
        }

        public Builder setRows(Iterable<RowToInsert> rows)
        {
            this.rows = Lists.newLinkedList(checkNotNull(rows));
            return this;
        }

        public Builder addRow(RowToInsert rowToInsert)
        {
            checkNotNull(rowToInsert);
            if (rows == null)
            {
                rows = Lists.newArrayList();
            }
            rows.add(rowToInsert);
            return this;
        }

        public Builder addRow(String id, Map<String, ?> content)
        {
            addRow(new RowToInsert(id, content));
            return this;
        }

        public Builder addRow(Map<String, ?> content)
        {
            addRow(new RowToInsert(null, content));
            return this;
        }

        public Builder setIgnoreUnknownValues(boolean ignoreUnknownValues)
        {
            this.ignoreUnknownValues = ignoreUnknownValues;
            return this;
        }

        public Builder setSkipInvalidRows(boolean skipInvalidRows)
        {
            this.skipInvalidRows = skipInvalidRows;
            return this;
        }

        public Builder setTemplateSuffix(String templateSuffix)
        {
            this.templateSuffix = templateSuffix;
            return this;
        }

        public InsertAllRequest build()
        {
            return new InsertAllRequest(this);
        }
    }

    private InsertAllRequest(Builder builder)
    {
        this.table = checkNotNull(builder.table);
        this.rows = ImmutableList.copyOf(checkNotNull(builder.rows));
        this.ignoreUnknownValues = builder.ignoreUnknownValues;
        this.skipInvalidRows = builder.skipInvalidRows;
        this.templateSuffix = builder.templateSuffix;
    }

    public TableId getTable()
    {
        return table;
    }

    public List<RowToInsert> getRows()
    {
        return rows;
    }

    public Boolean ignoreUnknownValues()
    {
        return ignoreUnknownValues;
    }

    public Boolean skipInvalidRows()
    {
        return skipInvalidRows;
    }

    public String getTemplateSuffix()
    {
        return templateSuffix;
    }

    public static Builder newBuilder(TableId table)
    {
        return new Builder().setTable(table);
    }

    public static Builder newBuilder(TableId table, Iterable<RowToInsert> rows)
    {
        return newBuilder(table).setRows(rows);
    }

    public static Builder newBuilder(TableId table, RowToInsert... rows)
    {
        return newBuilder(table, ImmutableList.copyOf(rows));
    }

    public static Builder newBuilder(String datasetId, String tableId)
    {
        return new Builder().setTable(TableId.of(datasetId, tableId));
    }

    public static Builder newBuilder(String datasetId, String tableId, Iterable<RowToInsert> rows)
    {
        return newBuilder(TableId.of(datasetId, tableId), rows);
    }

    public static Builder newBuilder(String datasetId, String tableId, RowToInsert... rows)
    {
        return newBuilder(TableId.of(datasetId, tableId), rows);
    }

    public static Builder newBuilder(TableInfo tableInfo, Iterable<RowToInsert> rows)
    {
        return newBuilder(tableInfo.getTableId(), rows);
    }

    public static Builder newBuilder(TableInfo tableInfo, RowToInsert... rows)
    {
        return newBuilder(tableInfo.getTableId(), rows);
    }

    public static InsertAllRequest of(TableId tableId, Iterable<RowToInsert> rows)
    {
        return newBuilder(tableId, rows).build();
    }

    public static InsertAllRequest of(TableId tableId, RowToInsert... rows)
    {
        return newBuilder(tableId, rows).build();
    }

    public static InsertAllRequest of(String datasetId, String tableId, Iterable<RowToInsert> rows)
    {
        return newBuilder(datasetId, tableId, rows).build();
    }

    public static InsertAllRequest of(String datasetId, String tableId, RowToInsert... rows)
    {
        return newBuilder(datasetId, tableId, rows).build();
    }

    public static InsertAllRequest of(TableInfo tableInfo, Iterable<RowToInsert> rows) 
    {
        return newBuilder(tableInfo.getTableId(), rows).build();
    }

    public static InsertAllRequest of(TableInfo tableInfo, RowToInsert... rows)
    {
        return newBuilder(tableInfo.getTableId(), rows).build();
    }

    @Override
    public String toString()
    {
        return MoreObjects.toStringHelper(this)
            .add("table", table)
            .add("rows", rows)
            .add("ignoreUnknownValues", ignoreUnknownValues)
            .add("skipInvalidRows", skipInvalidRows)
            .add("templateSuffix", templateSuffix)
            .toString();
    }

    @Override
    public int hashCode()
    {
        return Objects.hash(table, rows, ignoreUnknownValues, skipInvalidRows, templateSuffix);
    }

    @Override
    public boolean equals(Object obj)
    {
        if (obj == this)
        {
            return true;
        }
        if (!(obj instanceof InsertAllRequest))
        {
            return false;
        }
        InsertAllRequest other = (InsertAllRequest) obj;
        return Objects.equals(table, other.table)
            && Objects.equals(rows, other.rows)
            && Objects.equals(ignoreUnknownValues, other.ignoreUnknownValues)
            && Objects.equals(skipInvalidRows, other.skipInvalidRows)
            && Objects.equals(templateSuffix, other.templateSuffix);
    }
}
