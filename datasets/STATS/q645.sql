select  count(*) from comments as c,  		posts as p,  		postHistory as ph,          votes as v,          users as u  where u.Id = c.UserId 	and c.UserId = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = v.UserId  AND ph.CreationDate>='2010-08-20 08:25:44'::timestamp  AND p.Score=1  AND p.ViewCount=0  AND u.DownVotes<=1  AND u.UpVotes>=0  AND u.UpVotes<=68  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-21 00:00:00'::timestamp  AND v.CreationDate<='2014-09-12 00:00:00'::timestamp;