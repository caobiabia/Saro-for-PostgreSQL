select  count(*) from comments as c,  		posts as p,  		postHistory as ph,          votes as v,          users as u  where u.Id = c.UserId 	and c.UserId = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = v.UserId  AND c.Score=3  AND c.CreationDate>='2010-07-19 21:54:08'::timestamp  AND c.CreationDate<='2014-09-05 19:14:35'::timestamp  AND ph.PostHistoryTypeId=24  AND ph.CreationDate>='2011-02-10 09:46:07'::timestamp  AND p.PostTypeId=1  AND p.ViewCount>=0  AND p.CommentCount=0  AND u.Reputation<=735  AND u.Views>=0  AND u.DownVotes<=5  AND u.UpVotes<=18  AND v.VoteTypeId=2  AND v.CreationDate>='2010-07-29 00:00:00'::timestamp;