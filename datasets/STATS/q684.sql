select  count(*) from comments as c,  		postLinks as pl,  		postHistory as ph,          votes as v where pl.PostId = c.PostId 	and c.PostId = ph.PostId 	and ph.PostId = v.PostId  AND ph.CreationDate>='2010-10-28 18:33:14'::timestamp  AND ph.CreationDate<='2014-06-18 22:12:35'::timestamp  AND v.CreationDate>='2010-07-19 00:00:00'::timestamp  AND v.CreationDate<='2014-09-12 00:00:00'::timestamp;